from typing import Dict, Any, Union, List, Optional
from .base_agent import BaseAgent, AgentResponse
from core.routing_models import EMICalculation, EMIScenarios, LoanInputValidation
from core.telemetry_decorator import track_agent
from pydantic import ValidationError
import json


class WhatIfCalculator(BaseAgent):
    """Agent for calculating loan EMIs with different scenarios and fallback mechanisms."""

    def __init__(self):
        """Initialize WhatIfCalculator with all resources."""
        super().__init__("what_if_calculator", temperature=0.2)

        # Create structured LLM for EMI calculations
        self.structured_llm = self.llm.with_structured_output(EMIScenarios)

    def _validate_inputs(
        self,
        loan_amount: float,
        interest_rate: float,
        tenure_months: Optional[int] = None,
        prepayment: Optional[float] = None,
        outstanding_balance: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate calculation inputs using Pydantic model.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use Pydantic model for validation
            LoanInputValidation(
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                tenure_months=tenure_months,
                prepayment=prepayment,
                outstanding_balance=outstanding_balance,
            )
            return True, None
        except ValidationError as e:
            # Format validation errors into user-friendly message
            error_message = "⚠️ **Input Validation Error**\n\n**Issues Found:**\n"
            for error in e.errors():
                field = error["loc"][0]
                msg = error["msg"]
                error_message += f"• {field}: {msg}\n"

            error_message += "\n**Valid Ranges:**\n"
            error_message += "• Loan Amount: ₹1,000 to ₹10,00,00,000 (₹10 crore)\n"
            error_message += "• Interest Rate: 0.1% to 50%\n"
            error_message += "• Tenure: 6 to 360 months (0.5 to 30 years)\n"
            error_message += (
                "• Prepayment: Must not exceed loan amount or outstanding balance\n"
            )

            return False, error_message

    @track_agent("what_if_calculator")
    def process(
        self, query: Union[str, Dict[Any, Any]], retry_count: int = 0
    ) -> AgentResponse:
        """
        Process the loan calculation request with flexible input handling.

        Args:
            query: String query, dict with query/customer_data, or legacy dict formats

        Returns:
            AgentResponse: EMI calculations for different tenures
        """
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                return self._process_workflow_input(query)

            # Handle legacy input formats
            elif isinstance(query, dict):
                return self._process_legacy_dict(query)

            # Handle direct string queries
            else:
                return self._process_string_query(query)

        except Exception as e:
            return self.handle_error(f"What-if calculation error: {str(e)}")

    def _process_workflow_input(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process input from WorkflowEngine with query and optional customer_data list"""
        query = input_data.get("query", "")
        customer_data = input_data.get("customer_data", [])  # Always a list now

        # Extract parameters for validation if available
        loan_amount = None
        interest_rate = None
        outstanding_balance = None
        prepayment_amount = None

        # Get loan details from customer data if available
        if customer_data and len(customer_data) > 0:
            customer = customer_data[0]  # Use first customer for validation
            loan_amount = customer.get("loan_amount")
            interest_rate = customer.get("interest_rate")

            # Calculate outstanding balance if available
            if "loan_amount" in customer and "amount_paid" in customer:
                outstanding_balance = customer.get("loan_amount") - customer.get(
                    "amount_paid", 0
                )

        # Try to extract prepayment from query
        import re

        prepay_match = re.search(
            r"prepay(?:ment)?\s+(?:of\s+)?₹?(\d+(?:,\d+)*(?:\.\d+)?)",
            query,
            re.IGNORECASE,
        )
        if prepay_match:
            prepayment_str = prepay_match.group(1).replace(",", "")
            try:
                prepayment_amount = float(prepayment_str)
            except ValueError:
                pass

        # Validate inputs if we have them
        if loan_amount and interest_rate:
            is_valid, error_message = self._validate_inputs(
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                prepayment=prepayment_amount,
                outstanding_balance=outstanding_balance,
            )

            if not is_valid:
                print(f"❌ What-If Calculator validation failed")
                return AgentResponse(
                    answer=error_message,
                    metadata={},
                )

        # Enhanced prompt - let LLM handle 0, 1, or multiple customers intelligently
        calculation_prompt = self.prompts["customer_data_calculation"].format(
            query=query, customer_data=customer_data
        )

        # Use structured LLM to get EMI scenarios - LLM handles everything!
        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(calculation_prompt)
            return self._format_emi_response(emi_scenarios, customer_data, query)
        except ValidationError as e:
            return self.handle_error(f"EMI calculation validation error: {str(e)}")

    def _format_emi_response(
        self,
        emi_scenarios: EMIScenarios,
        customer_data: List[Dict] = None,
        original_query: str = "",
    ) -> AgentResponse:
        """Format EMI response with optional customer context - handles multiple customers"""
        answer = ""

        # Add customer context if available - LLM-friendly approach
        if customer_data:
            if len(customer_data) == 1:
                # Single customer
                customer = customer_data[0]
                answer += f"""Customer: {customer.get('customer_id')}
Current Loan: ₹{customer.get('loan_amount', 0):,.0f} at {customer.get('interest_rate', 0)}%
Current EMI: ₹{customer.get('monthly_emi', 0):,.0f}

"""
                # Add context about the request
                if (
                    "decrease" in original_query.lower()
                    or "reduce" in original_query.lower()
                ):
                    answer += "🎯 Request: Decrease current EMI\n\n"
                elif "increase" in original_query.lower():
                    answer += "🎯 Request: Increase loan amount\n\n"

            elif len(customer_data) > 1:
                # Multiple customers
                answer += "👥 Multiple Customer Analysis:\n"
                for customer in customer_data:
                    answer += f"• {customer.get('customer_id')}: ₹{customer.get('loan_amount', 0):,.0f} at {customer.get('interest_rate', 0)}%\n"
                answer += "\n"

        # Add EMI scenarios
        answer += f"EMI Calculation Results:\n\n"

        scenarios = emi_scenarios.scenarios
        for i, scenario in enumerate(scenarios, 1):
            years = scenario.tenure_months // 12
            answer += f"{i}. {years} Year(s) ({scenario.tenure_months} months):\n"
            answer += f"   • Monthly EMI: ₹{scenario.monthly_emi:,.0f}\n"
            answer += f"   • Total Interest: ₹{scenario.total_interest:,.0f}\n"
            answer += f"   • Total Amount: ₹{scenario.total_amount:,.0f}\n\n"

        # Only show summary and recommendation if multiple scenarios are present
        if len(scenarios) > 1:
            answer += f"Summary: {emi_scenarios.summary}\n"
            answer += f"Recommendation: {emi_scenarios.recommendation}"

        return AgentResponse(
            answer=answer,
            metadata={
                "scenarios": [scenario.dict() for scenario in scenarios],
            },
        )

    def _process_legacy_dict(self, query: Dict[Any, Any]) -> AgentResponse:
        """Handle legacy dictionary formats for backward compatibility"""
        # Check if it's routing data from supervisor
        if "parameters" in query and "confidence" in query:
            parameters = query["parameters"]
            loan_amount = parameters.get("loan_amount")
            interest_rate = parameters.get("interest_rate")

            if not loan_amount or not interest_rate:
                return self.handle_error(
                    "Missing required parameters: loan_amount or interest_rate"
                )

        # Direct parameter format
        elif "loan_amount" in query and "interest_rate" in query:
            loan_amount = query.get("loan_amount")
            interest_rate = query.get("interest_rate")
        else:
            return self.handle_error("Invalid legacy input format")

        # Process with structured parameters
        return self._calculate_emi_scenarios(loan_amount, interest_rate)

    def _process_string_query(self, query: str) -> AgentResponse:
        """Process natural language string queries"""
        prompt = self.prompts["natural_language_calculation"].format(query=query)

        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(prompt)
            return self._format_emi_response(
                emi_scenarios, [], query
            )  # Pass empty list for no customer data
        except ValidationError as e:
            return self.handle_error(
                f"Natural language EMI calculation error: {str(e)}"
            )

    def _calculate_emi_scenarios(
        self, loan_amount: float, interest_rate: float
    ) -> AgentResponse:
        """Calculate EMI scenarios for given loan amount and interest rate"""
        system_prompt = self.prompts["system"]
        user_prompt = (
            f"Calculate EMI scenarios for a loan amount of ₹{loan_amount:,.0f} "
            f"with an interest rate of {interest_rate}% per annum. "
            f"Provide structured calculations for tenures of 5, 10, 15, 20, and 30 years."
        )

        # Call structured LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Get structured EMI scenarios
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(messages)

            # Format structured response
            answer = f"EMI Calculation Results for ₹{loan_amount:,.0f} at {interest_rate}% annual interest:\n\n"

            for i, scenario in enumerate(emi_scenarios.scenarios, 1):
                years = scenario.tenure_months // 12
                answer += f"{i}. {years} Year(s) ({scenario.tenure_months} months):\n"
                answer += f"   • Monthly EMI: ₹{scenario.monthly_emi:,.0f}\n"
                answer += f"   • Total Interest: ₹{scenario.total_interest:,.0f}\n"
                answer += f"   • Total Amount: ₹{scenario.total_amount:,.0f}\n\n"

            answer += f"Summary: {emi_scenarios.summary}\n"
            answer += f"Recommendation: {emi_scenarios.recommendation}"

            return AgentResponse(
                answer=answer,
                metadata={
                    "scenarios": [
                        scenario.dict() for scenario in emi_scenarios.scenarios
                    ],
                },
            )

        except ValidationError as e:
            return self.handle_error(f"Invalid EMI calculation structure: {e}")
        except Exception as e:
            return self.handle_error(f"EMI calculation failed: {e}")
