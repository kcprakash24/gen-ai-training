"""
Integration tests - Critical Business Queries
Tests for specific customer questions mentioned in requirements
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


class TestCriticalCustomerQueries:
    """Test real customer questions from business requirements"""

    def test_how_many_emis_left_real_customer(self):
        """Critical: 'How many EMIs left?' - Most common query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Real customer query
        response = supervisor.process(
            "How many EMIs are left for customer 1900?", session_id="business-test-1"
        )

        # Should get specific answer
        assert response.answer is not None
        assert len(response.answer) > 20

        # Should mention EMIs or installments
        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["emi", "installment", "payment", "remaining", "left", "month"]
        )

        print(f"✅ EMIs Left Query Response:")
        print(f"   {response.answer[:300]}...")

    def test_topup_eligibility_with_good_history(self):
        """Critical: 'Am I eligible for a top-up?' - Key customer need"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Check top-up eligibility
        response = supervisor.process(
            "Am I eligible for a top-up loan for customer 1900?",
            session_id="business-test-2",
        )

        # Should get clear answer
        assert response.answer is not None

        # Should mention eligibility or top-up
        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["eligible", "topup", "top-up", "qualify", "additional"]
        )

        print(f"✅ Top-up Eligibility Response:")
        print(f"   {response.answer[:300]}...")

    def test_topup_eligibility_with_poor_history(self):
        """Test top-up eligibility for customer with issues"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for potentially ineligible customer
        response = supervisor.process(
            "Can customer 1902 get a top-up loan?", session_id="business-test-3"
        )

        # Should provide answer about eligibility
        assert response.answer is not None

        print(f"✅ Top-up Check (Different Customer):")
        print(f"   {response.answer[:300]}...")

    def test_prepayment_impact_on_emi(self):
        """Critical: 'If I prepay ₹10,000, how will my EMI change?'"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Prepayment scenario query
        response = supervisor.process(
            "If I prepay 10000 rupees, how will my EMI change?",
            session_id="business-test-4",
        )

        # Should discuss prepayment and EMI
        assert response.answer is not None
        assert len(response.answer) > 50

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["prepay", "emi", "change", "reduce", "impact"]
        )

        print(f"✅ Prepayment Impact Response:")
        print(f"   {response.answer[:300]}...")

    def test_prepayment_impact_on_tenure(self):
        """Test prepayment impact on loan tenure"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Prepayment with tenure focus
        response = supervisor.process(
            "If I prepay 50000, how much will my tenure reduce?",
            session_id="business-test-5",
        )

        # Should discuss tenure reduction
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["tenure", "month", "year", "period", "reduce", "shorter"]
        )

        print(f"✅ Tenure Reduction Response:")
        print(f"   {response.answer[:300]}...")

    def test_outstanding_balance_calculation(self):
        """Test outstanding balance query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Outstanding balance query
        response = supervisor.process(
            "What is the outstanding balance for customer 1900?",
            session_id="business-test-6",
        )

        # Should provide balance information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["balance", "outstanding", "remaining", "amount", "loan"]
        )

        print(f"✅ Outstanding Balance Response:")
        print(f"   {response.answer[:300]}...")

    def test_amount_paid_verification(self):
        """Test 'How much have I paid so far?' query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Amount paid query
        response = supervisor.process(
            "How much has customer 1900 paid so far?", session_id="business-test-7"
        )

        # Should provide payment information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(word in answer_lower for word in ["paid", "payment", "amount"])

        print(f"✅ Amount Paid Response:")
        print(f"   {response.answer[:300]}...")

    def test_next_due_date_query(self):
        """Test next EMI due date query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Next due date query
        response = supervisor.process(
            "When is the next EMI due for customer 1900?", session_id="business-test-8"
        )

        # Should provide due date information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower for word in ["due", "next", "date", "emi", "payment"]
        )

        print(f"✅ Next Due Date Response:")
        print(f"   {response.answer[:300]}...")

    def test_interest_saved_on_prepayment(self):
        """Test interest savings calculation on prepayment"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Interest savings query
        response = supervisor.process(
            "How much interest will I save if I prepay 100000?",
            session_id="business-test-9",
        )

        # Should discuss interest savings
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["interest", "save", "saving", "reduce", "prepay"]
        )

        print(f"✅ Interest Savings Response:")
        print(f"   {response.answer[:300]}...")

    def test_early_closure_charges(self):
        """Test early loan closure charges query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Early closure charges
        response = supervisor.process(
            "What are the charges for closing my loan early?",
            session_id="business-test-10",
        )

        # Should provide policy information on charges
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["charge", "closure", "foreclosure", "early", "penalty", "fee"]
        )

        print(f"✅ Early Closure Charges Response:")
        print(f"   {response.answer[:300]}...")


class TestLoanStatusQueries:
    """Test queries about loan status and details"""

    def test_loan_status_check(self):
        """Test current loan status query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Loan status query
        response = supervisor.process(
            "What is the current status of my loan for customer 1900?",
            session_id="status-test-1",
        )

        # Should provide status information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower for word in ["status", "active", "loan", "current"]
        )

        print(f"✅ Loan Status Response:")
        print(f"   {response.answer[:300]}...")

    def test_loan_details_summary(self):
        """Test comprehensive loan details query"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Full details query
        response = supervisor.process(
            "Show me all loan details for customer 1900", session_id="status-test-2"
        )

        # Should provide comprehensive information
        assert response.answer is not None
        assert len(response.answer) > 100  # Should be detailed

        answer_lower = response.answer.lower()
        # Should contain multiple loan-related terms
        loan_terms = ["loan", "amount", "emi", "interest", "tenure", "status"]
        matches = sum(1 for term in loan_terms if term in answer_lower)
        assert matches >= 3, f"Should mention at least 3 loan terms, found {matches}"

        print(f"✅ Loan Details Summary Response ({len(response.answer)} chars):")
        print(f"   {response.answer[:400]}...")

    def test_interest_rate_query(self):
        """Test interest rate inquiry"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Interest rate query
        response = supervisor.process(
            "What is the interest rate on my loan for customer 1900?",
            session_id="status-test-3",
        )

        # Should provide interest rate
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower for word in ["interest", "rate", "%", "percent"]
        )

        print(f"✅ Interest Rate Response:")
        print(f"   {response.answer[:300]}...")


# Test execution summary
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BUSINESS QUERIES - CRITICAL CUSTOMER QUESTIONS")
    print("=" * 80)
    print("\nThese tests validate the core business queries from requirements:")
    print("  1. How many EMIs left?")
    print("  2. Am I eligible for top-up?")
    print("  3. Prepayment impact scenarios")
    print("  4. Outstanding balance & amount paid")
    print("  5. Interest savings & early closure")
    print("\nTo run: pytest tests/integration/test_business_queries.py -v -s")
    print("=" * 80 + "\n")
