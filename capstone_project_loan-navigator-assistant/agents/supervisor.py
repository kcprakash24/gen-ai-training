"""
Simple Supervisor Agent following KISS, DRY, SOLID principles.
Pure routing with workflow delegation - no complex multi-step logic here.
"""

from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse
from core.factory import get_workflow_engine
from core.routing_models import RoutingDecision, AgentType
from core.telemetry_decorator import track_agent


class Supervisor(BaseAgent):
    """Simple supervisor - pure routing with workflow delegation following KISS principle."""

    def __init__(self):
        super().__init__("supervisor", temperature=0.1)
        self.workflow_engine = get_workflow_engine()
        self.structured_llm = self.llm.with_structured_output(RoutingDecision)

    @track_agent("supervisor")
    def process(
        self,
        query: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Simple routing with comprehensive decision-making and retry logic - KISS principle"""
        try:
            # Single LLM call for complete routing decision
            routing_decision = self._intelligent_route(query, context)

            print(
                f"🧠 Supervisor: {routing_decision.agent.value} (confidence: {routing_decision.confidence:.2f})"
            )
            print(f"🎯 Reasoning: {routing_decision.reasoning}")
            print(f"📊 Customer data needed: {routing_decision.needs_customer_data}")

            # FALLBACK 1: Detect ambiguous queries (confidence < 0.5)
            if routing_decision.confidence < 0.5:
                print("⚠️ Ambiguous query detected - requesting clarification")
                return self._handle_ambiguous_query(query, routing_decision, context)

            # FALLBACK 2: Handle low confidence (0.5 <= confidence < 0.7)
            if routing_decision.confidence < 0.7:
                print("⚠️ Low confidence routing - checking for multi-domain query")
                multi_domain_result = self._detect_multi_domain_query(query, context)
                if multi_domain_result["is_multi_domain"]:
                    print(
                        f"🔀 Multi-domain query detected: {multi_domain_result['domains']}"
                    )
                    return self._handle_multi_domain_query(
                        query, multi_domain_result, context, session_id
                    )
                # Low confidence, not multi-domain - request clarification
                return self._handle_ambiguous_query(query, routing_decision, context)

            # Clean delegation with comprehensive routing info - pass context and session_id
            response = self.workflow_engine.execute_workflow(
                query,
                routing_decision.agent,
                routing_decision.needs_customer_data,
                context,
                session_id,
            )

            # Check if Policy Guru needs query enhancement (fallback mechanism)
            if (
                routing_decision.agent == AgentType.POLICY_GURU
                and response.metadata
                and response.metadata.get("needs_query_enhancement")
            ):

                print(
                    "🔄 Policy Guru needs query enhancement - retrying with enriched context"
                )

                # Enhance query with additional context
                enhanced_query = self._enhance_query_with_context(
                    query, context, session_id
                )

                # Retry with enhanced query
                retry_response = self.workflow_engine.execute_workflow(
                    enhanced_query,
                    AgentType.POLICY_GURU,
                    routing_decision.needs_customer_data,
                    context,
                    session_id,
                    retry_count=1,
                )

                print(
                    f"✅ Retry complete - fallback: {retry_response.metadata.get('is_fallback', False)}"
                )
                return retry_response

            # Check if SQL Agent needs clarification (fallback mechanism)
            if (
                routing_decision.agent == AgentType.SQL_AGENT
                and response.metadata
                and response.metadata.get("sql_fallback_flag")
            ):
                print("🔄 SQL Agent needs clarification - returning guidance to user")
                return response

            return response

        except Exception as e:
            return self.handle_error(f"Supervisor error: {str(e)}")

    def _enhance_query_with_context(
        self, query: str, context: Optional[str], session_id: Optional[str]
    ) -> str:
        """Enhance query with additional context for retry attempts."""
        enhancement_prompt = self.prompts["query_enhancement"].format(
            query=query, context=context if context else "No conversation history"
        )

        try:
            enhanced = self.llm.invoke(enhancement_prompt).content.strip()
            print(f"📝 Enhanced query: {enhanced[:100]}...")
            return enhanced
        except:
            # If enhancement fails, return original with context appended
            if context:
                return f"{query}\n\nAdditional Context: {context}"
            return query

    def _intelligent_route(
        self, query: str, context: Optional[str] = None
    ) -> RoutingDecision:
        """Single LLM call for comprehensive routing decision - DRY principle"""
        routing_prompt = self.prompts["intelligent_routing_prompt"]
        full_prompt = f"{routing_prompt}\n\nUser Query: {query}\n{f'Available context: {context}' if context else 'No additional context available.'}"
        return self.structured_llm.invoke(full_prompt)

    def _handle_ambiguous_query(
        self, query: str, routing_decision: RoutingDecision, context: Optional[str]
    ) -> AgentResponse:
        """Handle ambiguous queries by requesting clarification from user"""
        clarification_prompt = self.prompts["ambiguous_query_clarification"]

        formatted_prompt = clarification_prompt.format(
            query=query,
            reasoning=routing_decision.reasoning,
            confidence=routing_decision.confidence,
        )

        response = self.llm.invoke(formatted_prompt).content

        return AgentResponse(
            answer=response,
            metadata={
                "requires_clarification": True,
            },
        )

    def _detect_multi_domain_query(self, query: str, context: Optional[str]) -> dict:
        """Detect if query spans multiple domains (SQL + Calculator + Policy)"""
        detection_prompt = self.prompts["multi_domain_detection"]

        formatted_prompt = detection_prompt.format(
            query=query, context=context if context else "No context"
        )

        try:
            response = self.llm.invoke(formatted_prompt).content
            # Try to parse JSON response
            import json

            if "{" in response and "}" in response:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                return json.loads(json_str)
        except:
            pass

        # Default: not multi-domain
        return {
            "is_multi_domain": False,
            "domains": [],
            "reasoning": "Single domain query",
        }

    def _handle_multi_domain_query(
        self,
        query: str,
        multi_domain_result: dict,
        context: Optional[str],
        session_id: Optional[str],
    ) -> AgentResponse:
        """Handle queries that span multiple domains by coordinating agents"""
        domains = multi_domain_result.get("domains", [])

        print(f"🔀 Processing multi-domain query across: {', '.join(domains)}")

        # Execute each domain in sequence and combine results
        results = []

        for domain in domains:
            try:
                agent_type = AgentType(domain)
                print(f"  → Executing {domain}...")

                response = self.workflow_engine.execute_workflow(
                    query, agent_type, True, context, session_id
                )

                results.append(
                    {
                        "agent": domain,
                        "answer": response.answer,
                        "metadata": response.metadata,
                    }
                )

            except Exception as e:
                print(f"  ❌ Error executing {domain}: {str(e)}")
                continue

        # Combine results into coherent response
        combined_answer = self._combine_multi_domain_results(query, results)

        return AgentResponse(answer=combined_answer)

    def _combine_multi_domain_results(self, query: str, results: list) -> str:
        """Combine multiple agent results into a coherent response"""
        combination_prompt = self.prompts["multi_domain_combination"]

        results_text = "\n\n".join(
            [f"**{r['agent']}**:\n{r['answer']}" for r in results]
        )

        formatted_prompt = combination_prompt.format(query=query, results=results_text)

        try:
            return self.llm.invoke(formatted_prompt).content
        except:
            # Fallback: just concatenate results
            return "\n\n---\n\n".join([r["answer"] for r in results])
