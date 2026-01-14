"""
Simple SQL Agent for converting natural language queries to SQL and fetching loan data.
"""

from typing import Dict, Any, List, Union, Optional
import sqlite3
import json
import re

from .base_agent import BaseAgent, AgentResponse
from core.routing_models import SQLQueryResult
from core.constants import LOAN_DB_PATH
from core.telemetry_decorator import track_agent
from pydantic import ValidationError


class SQLAgent(BaseAgent):
    """Simple agent for converting NLP to SQL and fetching loan data."""

    def __init__(self):
        """Initialize SQLAgent."""
        super().__init__("sql_agent", temperature=0.1)
        self.db_path = LOAN_DB_PATH

        # Create structured LLM for SQL query generation
        self.structured_llm = self.llm.with_structured_output(SQLQueryResult)

        # Known valid table names for validation
        self.valid_tables = ["loan_data"]

    def _generate_sql(self, query: str) -> SQLQueryResult:
        """Generate SQL query from natural language with structured output."""
        try:
            # Use self.prompts for structured query generation
            prompt = self.prompts["sql_prompt"].format(user_query=query)

            system_prompt = self.prompts["system_prompt"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Get structured SQL query response
            return self.structured_llm.invoke(messages)

        except ValidationError as e:
            raise ValueError(f"Invalid SQL query structure: {e}")

    def _execute_sql_with_params(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        """
        Execute parameterized SQL query with parameters (test utility).

        Used for testing SQL injection prevention with parameterized queries.
        Production code uses _execute_sql() with LLM-generated queries.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = [dict(row) for row in rows]
            conn.close()

            return results
        except Exception as e:
            if conn:
                conn.close()
            raise e

    def _validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query before execution.
        Returns validation result with is_valid flag and issues list.
        """
        validation_result = {"is_valid": True, "issues": [], "warnings": []}

        sql_upper = sql_query.upper().strip()

        # 1. Security check: only allow SELECT statements
        if not sql_upper.startswith(("SELECT", "WITH")):
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                "Only SELECT and WITH queries are allowed"
            )
            return validation_result

        # 2. Check for dangerous SQL operations
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "TRUNCATE",
            "EXEC",
        ]
        for keyword in dangerous_keywords:
            if re.search(rf"\b{keyword}\b", sql_upper):
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Dangerous SQL keyword detected: {keyword}"
                )

        # 3. Validate table names
        sql_lower = sql_query.lower()
        for table in self.valid_tables:
            if table in sql_lower:
                break
        else:
            # No valid table found
            validation_result["warnings"].append(
                "Query might reference invalid table name"
            )

        # 4. Check for basic SQL syntax patterns
        if "SELECT" in sql_upper and "FROM" not in sql_upper:
            validation_result["is_valid"] = False
            validation_result["issues"].append("SELECT query must have FROM clause")

        # 5. Check for SQL injection patterns
        injection_patterns = [r"';", r"--", r"/\*", r"\*/", r"xp_", r"sp_"]
        for pattern in injection_patterns:
            if re.search(pattern, sql_query):
                validation_result["warnings"].append(
                    f"Potential SQL injection pattern detected: {pattern}"
                )

        return validation_result

    def _execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results after validation."""
        # Validate SQL before execution
        validation = self._validate_sql(sql_query)

        if not validation["is_valid"]:
            raise ValueError(
                f"SQL validation failed: {', '.join(validation['issues'])}"
            )

        # Log warnings if any
        if validation["warnings"]:
            print(f"⚠️ SQL Warnings: {', '.join(validation['warnings'])}")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def _format_results_with_llm(
        self, query: str, results: List[Dict[str, Any]], sql_explanation: str
    ) -> str:
        """Use LLM to format SQL results in a natural, user-friendly way."""
        if not results:
            return "No data found matching your criteria."

        # Prepare data summary
        data_json = json.dumps(results, indent=2, default=str)

        # Use prompt from YAML configuration
        format_prompt = self.prompts["format_prompt"].format(
            user_query=query,
            sql_explanation=sql_explanation,
            num_records=len(results),
            data_json=data_json,
        )

        response = self.llm.invoke(format_prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _generate_clarification_prompt(
        self, original_query: str, error_type: str
    ) -> str:
        """Generate guided clarification prompt for failed queries."""
        # Map error types to YAML prompt keys
        prompt_keys = {
            "zero_results": "zero_results_clarification",
            "validation_failed": "validation_failed_clarification",
            "sql_error": "sql_error_clarification",
        }

        # Get the appropriate prompt from YAML
        prompt_key = prompt_keys.get(error_type, "sql_error_clarification")
        clarification_template = self.prompts[prompt_key]

        # Format the prompt with the original query
        return clarification_template.format(original_query=original_query)

    @track_agent("sql_agent")
    def process(
        self, query: Union[str, Dict[str, Any]], retry_count: int = 0
    ) -> AgentResponse:
        """Process NLP query to fetch loan data with fallback mechanisms."""
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                nlp_query = query.get("query", "")
                customer_data = query.get("customer_data", [])  # Always a list now
                context = query.get("context", "")

                # Add customer context to query if available
                if customer_data:
                    customer_ids = [
                        str(c.get("customer_id", ""))
                        for c in customer_data
                        if c.get("customer_id") is not None
                    ]
                    if customer_ids:
                        nlp_query = (
                            f"{nlp_query} [Customer IDs: {', '.join(customer_ids)}]"
                        )
            else:
                # Handle legacy input formats
                if isinstance(query, dict):
                    nlp_query = query.get("query", str(query))
                else:
                    nlp_query = query
                customer_data = []
                context = ""

            if not nlp_query:
                return self.handle_error("Empty query provided")

            # Generate structured SQL query
            sql_result = self._generate_sql(nlp_query)

            # Validate SQL before execution
            validation = self._validate_sql(sql_result.query)

            if not validation["is_valid"]:
                # SQL validation failed - signal supervisor for clarification
                print(f"❌ SQL Validation Failed: {validation['issues']}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, "validation_failed"
                )

                if retry_count == 0:
                    # First attempt - signal supervisor
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "retry_count": retry_count,
                        },
                    )
                else:
                    # Retry also failed - return clarification directly
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "is_fallback": True,
                            "retry_count": retry_count,
                        },
                    )

            # Execute validated SQL
            try:
                results = self._execute_sql(sql_result.query)
            except Exception as exec_error:
                print(f"❌ SQL Execution Error: {exec_error}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, "sql_error"
                )

                if retry_count == 0:
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "retry_count": retry_count,
                        },
                    )
                else:
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "is_fallback": True,
                            "retry_count": retry_count,
                        },
                    )

            # Zero-result detection
            if len(results) == 0:
                print(f"⚠️ Zero Results Returned for query: {nlp_query}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, "zero_results"
                )

                if retry_count == 0 and not customer_data:
                    # No customer data and no results - signal supervisor for clarification
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "retry_count": retry_count,
                        },
                    )
                else:
                    # Return friendly zero-results message
                    answer = f"No data found matching your criteria.\n\n{clarification}"
                    return AgentResponse(
                        answer=answer,
                        metadata={
                            "retry_count": retry_count,
                        },
                    )

            # Format response using LLM for natural presentation
            answer = self._format_results_with_llm(
                query=nlp_query if isinstance(query, str) else query.get("query", ""),
                results=results,
                sql_explanation=sql_result.explanation,
            )

            return AgentResponse(
                answer=answer,
                sources=[
                    {
                        "type": "database",
                        "sql": sql_result.query,
                        "explanation": sql_result.explanation,
                        "confidence": sql_result.confidence,
                        "count": len(results),
                    }
                ],
                metadata={
                    "retry_count": retry_count,
                },
            )

        except Exception as e:
            print(f"❌ Unexpected error in SQL Agent: {e}")
            return self.handle_error(f"SQL error: {str(e)}")

    def get_data(self, query: str) -> List[Dict[str, Any]]:
        """Simple method to get raw data for other agents."""
        try:
            sql_result = self._generate_sql(query)
            return self._execute_sql(sql_result.query)
        except Exception:
            return []
