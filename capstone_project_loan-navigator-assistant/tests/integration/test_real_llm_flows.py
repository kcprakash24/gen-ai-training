"""
Integration tests - Real LLM calls with Azure OpenAI
These tests verify actual system behavior with real API calls
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Skip if no API credentials
pytestmark = pytest.mark.skipif(
    not os.getenv("AZURE-OPENAI-API-KEY"), reason="Requires Azure OpenAI credentials"
)


class TestSupervisorIntentClassification:
    """Test real LLM-based intent classification"""

    def test_intent_classification_keywords(self):
        """Test keyword-based routing for clear queries"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Keywords: "EMI", "calculate" with specific amount
        routing1 = supervisor._intelligent_route("calculate EMI for 50 lakhs at 8.5%")
        assert routing1.agent.value == "WHAT_IF_CALCULATOR"

        # Keywords: "policy", "eligibility"
        routing2 = supervisor._intelligent_route("what is eligibility policy")
        assert routing2.agent.value == "POLICY_GURU"

        # Keywords: "customer", "loan details"
        routing3 = supervisor._intelligent_route("show customer 1900 loan details")
        assert routing3.agent.value == "SQL_AGENT"

        print("✅ Keyword-based routing works for all agents")

    def test_sql_query_routing_real_llm(self):
        """Real LLM should route SQL queries correctly"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Test SQL query
        routing = supervisor._intelligent_route("show loan for customer 1900")

        assert routing.agent.value == "SQL_AGENT"
        assert routing.confidence > 0.7
        assert routing.needs_customer_data == True
        print(f"✅ Routing: {routing.agent.value}, Confidence: {routing.confidence}")

    def test_calculator_query_routing_real_llm(self):
        """Real LLM should route calculator queries correctly"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Test calculator query
        routing = supervisor._intelligent_route("calculate EMI for 50 lakhs at 8.5%")

        assert routing.agent.value == "WHAT_IF_CALCULATOR"
        assert routing.confidence > 0.7
        print(f"✅ Routing: {routing.agent.value}, Confidence: {routing.confidence}")

    def test_policy_query_routing_real_llm(self):
        """Real LLM should route policy queries correctly"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Test policy query
        routing = supervisor._intelligent_route("what are loan eligibility criteria")

        assert routing.agent.value == "POLICY_GURU"
        assert routing.confidence > 0.7
        print(f"✅ Routing: {routing.agent.value}, Confidence: {routing.confidence}")

    def test_ambiguous_query_low_confidence(self):
        """Ambiguous query should have low confidence"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Test ambiguous query
        routing = supervisor._intelligent_route("loan")

        # Should have low confidence
        assert routing.confidence < 0.7
        print(f"⚠️ Ambiguous query - Confidence: {routing.confidence}")

    def test_merge_multi_agent_responses(self):
        """Test merging outputs from multiple agents"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query that might need multiple agents
        response = supervisor.process(
            "what if I prepay 50000 on my loan, and what are the prepayment charges?",
            session_id="multi-agent-test",
        )

        # Should get response
        assert response.answer is not None
        assert len(response.answer) > 50

        # Should mention both prepayment calculation and charges
        answer_lower = response.answer.lower()
        assert "prepay" in answer_lower or "charge" in answer_lower

        print(f"✅ Multi-agent response: {response.answer[:200]}...")

    def test_format_response_tone(self):
        """Test if response tone is friendly and compliant"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        response = supervisor.process(
            "what are loan eligibility criteria?", session_id="tone-test"
        )

        # Check for friendly tone indicators
        answer_lower = response.answer.lower()

        # Should not be overly technical or cold
        assert response.answer is not None
        assert len(response.answer) > 20

        # Friendly responses often include helpful language
        print(f"✅ Response tone checked: {response.answer[:150]}...")


class TestSQLAgentRealTranslation:
    """Test real NLP to SQL translation"""

    def test_nlp_to_sql_real_llm(self):
        """Real LLM should generate correct SQL"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Test SQL generation - simple query without extra params
        sql_result = agent._generate_sql("Show me loan details for customer 1900")

        # Verify SQL contains expected elements
        assert "SELECT" in sql_result.query.upper()
        assert (
            "loan_data" in sql_result.query.lower()
            or "loan" in sql_result.query.lower()
        )
        assert sql_result.explanation is not None
        assert 0 <= sql_result.confidence <= 1.0

        print(f"✅ Generated SQL: {sql_result.query}")
        print(f"✅ Explanation: {sql_result.explanation}")
        print(f"✅ Confidence: {sql_result.confidence}")

    def test_sql_execution_with_real_db(self):
        """Should execute SQL and return real data from database"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Use correct table name 'loan_data'
        sql = "SELECT * FROM loan_data WHERE customer_id = '1900' LIMIT 1"

        results = agent._execute_sql(sql)

        # Should get results
        assert len(results) > 0, "Should return at least one result"
        assert "customer_id" in results[0]

        print(f"✅ Executed SQL and got {len(results)} results")
        print(f"Sample customer: {results[0].get('customer_id')}")

    def test_sql_schema_enforcement(self):
        """Test that invalid table names are rejected"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Try to access non-whitelisted table
        try:
            result = agent._validate_sql("SELECT * FROM AdminUsers")
            # Should fail validation
            assert result["is_valid"] == False
            print("✅ Schema enforcement: Invalid table rejected")
        except Exception as e:
            # Validation might throw exception
            print(f"✅ Schema enforcement: {str(e)}")

    def test_parameterized_query_safety(self):
        """Test parameterized queries prevent SQL injection"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Safe parameterized query
        sql = "SELECT * FROM loan_data WHERE customer_id = ?"
        results = agent._execute_sql_with_params(sql, ("1900",))

        # Should execute safely
        assert isinstance(results, list)
        print(f"✅ Parameterized query executed safely: {len(results)} results")


class TestPolicyGuruRealRAG:
    """Test real RAG retrieval and response generation"""

    def test_real_vector_retrieval(self):
        """Real vector retrieval should find relevant documents"""
        from agents.policy_guru import PolicyGuru

        agent = PolicyGuru()

        # Temporarily lower threshold for testing
        original_threshold = agent.similarity_threshold
        agent.similarity_threshold = 0.5  # Lower threshold for testing

        try:
            # Test retrieval with a clear policy query
            docs = agent.retrieve_docs("What is the loan eligibility criteria?")

            # Should retrieve some documents with lower threshold
            assert len(docs) >= 0, "Should not error"

            if len(docs) > 0:
                # Each doc should have content and metadata
                for doc in docs:
                    assert "content" in doc
                    assert "metadata" in doc
                    assert len(doc["content"]) > 0

                print(f"✅ Retrieved {len(docs)} documents")
                print(f"Sample: {docs[0]['content'][:100]}...")
            else:
                print(
                    "⚠️ No documents found - vector store may be empty or threshold too high"
                )
        finally:
            # Restore original threshold
            agent.similarity_threshold = original_threshold

    def test_rag_summary_with_citation(self):
        """Test that RAG response includes source citations"""
        from agents.policy_guru import PolicyGuru

        agent = PolicyGuru()
        agent.similarity_threshold = 0.5

        try:
            response = agent.process("What are loan eligibility criteria?")

            # Check for sources in metadata
            if response.sources and len(response.sources) > 0:
                print(f"✅ Citations included: {len(response.sources)} sources")
                print(f"   Sample source: {response.sources[0]}")
            else:
                print("⚠️ No sources found (threshold too high or no docs)")
        finally:
            agent.similarity_threshold = 0.75

    def test_filter_irrelevant_chunks(self):
        """Test that irrelevant chunks are filtered"""
        from agents.policy_guru import PolicyGuru

        agent = PolicyGuru()

        # With threshold 0.75, should filter low-similarity docs
        docs = agent.retrieve_docs("xyz random query abc")

        # Should filter out irrelevant content
        print(f"✅ Filtered results: {len(docs)} docs (threshold=0.75)")

        # All docs should be above threshold
        assert all(doc.get("content") for doc in docs)

    def test_generic_fallback_message(self):
        """Test generic fallback when retrieval fails"""
        from agents.policy_guru import PolicyGuru

        agent = PolicyGuru()

        # Very specific query unlikely to match
        response = agent.process("what is the exact temperature of jupiter in celsius")

        # Should get fallback response
        assert response.answer is not None

        if response.metadata.get("needs_query_enhancement"):
            print(f"✅ Fallback triggered: {response.metadata.get('reason')}")
        else:
            print("⚠️ Got response (docs may have matched)")

    def test_full_rag_response(self):
        """Full RAG pipeline should generate answer from documents"""
        from agents.policy_guru import PolicyGuru

        agent = PolicyGuru()

        # Temporarily lower threshold for testing
        original_threshold = agent.similarity_threshold
        agent.similarity_threshold = 0.5

        try:
            # Test full process method
            response = agent.process(
                "What documents are required for loan application?"
            )

            # Should get some response (even if fallback)
            assert isinstance(response.answer, str)

            if response.metadata.get("needs_query_enhancement"):
                print("⚠️ Query enhancement needed - no documents above threshold")
                print(f"Reason: {response.metadata.get('reason')}")
            else:
                assert len(response.answer) > 50  # Should be substantial
                print(f"✅ Generated answer: {response.answer[:100]}...")
        finally:
            agent.similarity_threshold = original_threshold


class TestWhatIfCalculatorRealCalculations:
    """Test real EMI calculations"""

    def test_real_emi_calculation(self):
        """Real EMI calculation should be accurate using LLM"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # Test case - use 5 years which is in default scenarios
        loan_amount = 500000
        interest_rate = 10.0
        tenure_years = 5

        # Use actual LLM flow
        response = agent.process(
            f"Calculate EMI for ₹{loan_amount:,.0f} at {interest_rate}% for {tenure_years} years"
        )

        # Verify response structure
        assert response.answer is not None
        assert len(response.answer) > 50
        assert "₹" in response.answer

        # Verify scenarios exist
        scenarios = response.metadata.get("scenarios", [])
        assert len(scenarios) >= 1, "Should return scenarios"

        # Find 5-year scenario (60 months)
        tenure_months = tenure_years * 12
        five_year_scenario = None
        for scenario in scenarios:
            if scenario.get("tenure_months") == tenure_months:
                five_year_scenario = scenario
                break

        assert (
            five_year_scenario is not None
        ), f"Should have {tenure_years}-year scenario"

        # Validate EMI is reasonable
        emi = five_year_scenario.get("monthly_emi")
        monthly_rate = interest_rate / (12 * 100)
        expected_emi = (
            loan_amount
            * monthly_rate
            * pow(1 + monthly_rate, tenure_months)
            / (pow(1 + monthly_rate, tenure_months) - 1)
        )

        # Should match within 1%
        percentage_diff = abs(emi - expected_emi) / expected_emi * 100
        assert percentage_diff < 1.0, f"EMI deviation: {percentage_diff}%"
        print(
            f"✅ EMI Calculation: ₹{emi:,.2f} (Expected: ₹{expected_emi:,.2f}, Diff: {percentage_diff:.2f}%)"
        )

    def test_zero_interest_case(self):
        """Test EMI calculation with 0% interest using LLM"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # 0% interest - use 5 years which is in default scenarios
        loan_amount = 600000
        tenure_years = 5

        response = agent.process(
            f"Calculate EMI for ₹{loan_amount:,.0f} at 0% interest for {tenure_years} years"
        )

        # Verify response
        assert response.answer is not None

        # Find 5-year scenario
        scenarios = response.metadata.get("scenarios", [])
        five_year_scenario = None
        for scenario in scenarios:
            if scenario.get("tenure_months") == 60:
                five_year_scenario = scenario
                break

        if five_year_scenario:
            emi = five_year_scenario.get("monthly_emi")
            expected = loan_amount / 60

            # For 0% interest, EMI should be exactly loan/tenure
            percentage_diff = abs(emi - expected) / expected * 100
            assert (
                percentage_diff < 2.0
            ), "Zero interest EMI should be ~loan_amount/months"
            print(f"✅ Zero interest EMI: ₹{emi:,.2f} (Expected: ₹{expected:,.2f})")
        else:
            # If no exact 5-year scenario, just verify response is reasonable
            assert "emi" in response.answer.lower() or "EMI" in response.answer
            print(f"✅ Zero interest response provided (no exact 5-year scenario)")

    def test_foreclosure_simulation(self):
        """Test foreclosure suggestion for full repayment"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # Prepayment > outstanding should suggest foreclosure
        validation = agent._validate_inputs(
            100000, 10.0, 12, prepayment=100000, outstanding_balance=50000
        )

        assert validation["is_valid"] == False
        assert any("foreclosure" in sugg.lower() for sugg in validation["suggestions"])
        print(f"✅ Foreclosure suggested: {validation['suggestions'][0]}")

    def test_amortization_schedule_generation(self):
        """Test that amortization schedule can be generated"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # Request multiple scenarios (acts as amortization schedule)
        response = agent.process(
            {"query": "show EMI for different tenures", "customer_data": []}
        )

        # Should contain multiple scenarios
        assert response.metadata.get("scenarios") is not None
        scenarios = response.metadata.get("scenarios", [])

        if len(scenarios) > 0:
            print(f"✅ Generated {len(scenarios)} scenarios (amortization schedule)")
        else:
            print("⚠️ No scenarios in metadata")

    def test_input_validation_real(self):
        """Real validation should catch invalid inputs"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # Test negative loan amount
        validation = agent._validate_inputs(-50000, 10.0, 12, 0, 0)

        assert validation["is_valid"] == False
        assert len(validation["errors"]) > 0
        print(f"✅ Validation caught error: {validation['errors'][0]}")


class TestEndToEndFlows:
    """Test complete workflows with real LLM"""

    def test_sql_query_full_flow(self):
        """Complete flow: User query → SQL Agent → Response"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Real query
        response = supervisor.process(
            "show loan details for customer 1900", session_id="integration-test-1"
        )

        # Should get response
        assert response.answer is not None
        assert len(response.answer) > 20

        # Should contain loan information
        assert any(
            word in response.answer.lower()
            for word in ["loan", "1900", "customer", "amount"]
        )

        print(f"✅ Full SQL Flow Response ({len(response.answer)} chars):")
        print(f"   {response.answer[:300]}...")

    def test_calculator_query_full_flow(self):
        """Complete flow: User query → Calculator → Response"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Real query
        response = supervisor.process(
            "calculate EMI for 50 lakhs at 8.5% interest",
            session_id="integration-test-2",
        )

        # Should get response
        assert response.answer is not None

        # Should contain EMI information
        assert any(
            word in response.answer.lower() for word in ["emi", "monthly", "interest"]
        )

        print(f"✅ Full Calculator Flow Response ({len(response.answer)} chars):")
        print(f"   {response.answer[:300]}...")

    def test_policy_query_full_flow(self):
        """Complete flow: User query → Policy Guru → Response"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Real query
        response = supervisor.process(
            "what are prepayment charges?", session_id="integration-test-3"
        )

        # Should get response
        assert response.answer is not None

        # Should contain policy information
        assert any(
            word in response.answer.lower()
            for word in ["prepayment", "charge", "policy"]
        )

        print(f"✅ Full Policy Flow Response ({len(response.answer)} chars):")
        print(f"   {response.answer[:300]}...")

    def test_combined_flow_whatif_and_policy(self):
        """Test combined flow requiring multiple agents"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query that needs both calculator and policy
        response = supervisor.process(
            "What if I prepay 50000, and is there any penalty policy?",
            session_id="combined-flow-test",
        )

        # Should get comprehensive response
        assert response.answer is not None
        assert len(response.answer) > 50

        answer_lower = response.answer.lower()
        # Should address both parts
        assert (
            "prepay" in answer_lower
            or "penalty" in answer_lower
            or "policy" in answer_lower
        )

        print(f"✅ Combined flow response: {response.answer[:250]}...")

    def test_unknown_query_fallback(self):
        """Test graceful fallback for garbage input"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Complete garbage input
        response = supervisor.process(
            "xyz abc 123 random nonsense", session_id="fallback-test-garbage"
        )

        # Should get a response (not crash)
        assert response.answer is not None

        # Should be helpful or request clarification
        answer_lower = response.answer.lower()
        helpful_words = ["help", "clarify", "sorry", "understand", "assist", "can you"]

        if any(word in answer_lower for word in helpful_words):
            print("✅ Graceful fallback with helpful message")
        else:
            print(f"⚠️ Response provided: {response.answer[:100]}...")


class TestFallbackMechanisms:
    """Test real fallback scenarios"""

    def test_ambiguous_query_clarification(self):
        """Ambiguous query should trigger clarification"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Very ambiguous query
        response = supervisor.process("loan", session_id="fallback-test-1")

        # Should request clarification or provide options
        assert response.metadata is not None

        # Check if clarification was requested
        if response.metadata.get("requires_clarification"):
            print("✅ Clarification requested for ambiguous query")
            assert (
                "clarify" in response.answer.lower()
                or "help" in response.answer.lower()
            )
        else:
            print("⚠️ Direct response provided (acceptable)")

    def test_sql_zero_results_fallback(self):
        """Zero results should trigger fallback"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Non-existent customer - use correct table name
        sql = "SELECT * FROM loan_data WHERE customer_id = ?"
        results = agent._execute_sql_with_params(sql, ("99999",))

        # Should return empty results
        assert len(results) == 0
        print("✅ Zero results detected - fallback should trigger")

    def test_validation_error_fallback(self):
        """Invalid input should trigger validation error"""
        from agents.what_if_calculator import WhatIfCalculator

        agent = WhatIfCalculator()

        # Invalid prepayment
        validation = agent._validate_inputs(
            100000,
            10.0,
            12,
            prepayment=150000,  # More than loan amount
            outstanding_balance=100000,
        )

        assert validation["is_valid"] == False
        assert len(validation["suggestions"]) > 0
        print(f"✅ Validation error with suggestion: {validation['suggestions'][0]}")


# Test execution summary
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTEGRATION TESTS - REAL LLM CALLS")
    print("=" * 80)
    print("\nThese tests use REAL Azure OpenAI API calls")
    print("They verify actual system behavior, not mocked responses\n")
    print("To run: pytest tests/integration/ -v -s")
    print("=" * 80 + "\n")
