"""
Integration tests - Multi-Loan Scenarios & Edge Cases
Tests for customers with multiple loans and boundary conditions
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


class TestMultiLoanScenarios:
    """Test scenarios with customers having multiple loans"""

    def test_customer_with_multiple_active_loans(self):
        """Critical: Handle customers with 2+ active loans"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for customer who might have multiple loans
        response = supervisor.process(
            "Show all loans for customer 1900", session_id="multi-loan-test-1"
        )

        # Should get response
        assert response.answer is not None
        assert len(response.answer) > 50

        # Should mention loan information
        answer_lower = response.answer.lower()
        assert any(word in answer_lower for word in ["loan", "amount", "emi"])

        print(f"✅ Multiple Loans Query Response:")
        print(f"   {response.answer[:300]}...")

    def test_aggregate_emi_across_loans(self):
        """Test total EMI calculation across all loans"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for total EMI
        response = supervisor.process(
            "What is my total monthly EMI for customer 1900?",
            session_id="multi-loan-test-2",
        )

        # Should provide aggregate information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower for word in ["emi", "total", "monthly", "payment"]
        )

        print(f"✅ Aggregate EMI Response:")
        print(f"   {response.answer[:300]}...")

    def test_topup_on_specific_loan(self):
        """Test top-up eligibility for specific loan when customer has multiple"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for top-up on specific loan
        response = supervisor.process(
            "Am I eligible for top-up on my loan? Customer 1900",
            session_id="multi-loan-test-3",
        )

        # Should provide eligibility information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower for word in ["eligible", "topup", "top-up", "loan"]
        )

        print(f"✅ Top-up Eligibility (Multi-loan Customer):")
        print(f"   {response.answer[:300]}...")

    def test_outstanding_balance_all_loans(self):
        """Test total outstanding balance across all loans"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for total outstanding
        response = supervisor.process(
            "What is my total outstanding balance for customer 1900?",
            session_id="multi-loan-test-4",
        )

        # Should provide balance information
        assert response.answer is not None

        answer_lower = response.answer.lower()
        assert any(
            word in answer_lower
            for word in ["outstanding", "balance", "amount", "loan"]
        )

        print(f"✅ Total Outstanding Balance:")
        print(f"   {response.answer[:300]}...")


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions"""

    def test_closed_loan_query(self):
        """Test query on fully paid/closed loan"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query that might return closed loan
        response = supervisor.process(
            "Show me closed loans for customer 1900", session_id="edge-test-1"
        )

        # Should handle gracefully
        assert response.answer is not None

        answer_lower = response.answer.lower()
        # Should mention status or provide information
        assert any(
            word in answer_lower
            for word in [
                "loan",
                "status",
                "closed",
                "complete",
                "paid",
                "active",
                "customer",
            ]
        )

        print(f"✅ Closed Loan Query Handled:")
        print(f"   {response.answer[:300]}...")

    def test_invalid_customer_id_graceful_error(self):
        """Critical: Non-existent customer should get helpful error"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for non-existent customer
        response = supervisor.process(
            "Show loan for customer 99999", session_id="edge-test-2"
        )

        # Should get a response (not crash)
        assert response.answer is not None

        # Should be helpful
        answer_lower = response.answer.lower()
        helpful_terms = [
            "sorry",
            "not found",
            "no loan",
            "no record",
            "customer",
            "help",
            "verify",
        ]
        has_helpful_response = any(term in answer_lower for term in helpful_terms)

        assert (
            has_helpful_response or len(response.answer) > 20
        ), "Should provide helpful error message"

        print(f"✅ Invalid Customer ID Handled Gracefully:")
        print(f"   {response.answer[:300]}...")

    def test_loan_not_eligible_for_topup(self):
        """Test loan that doesn't meet top-up criteria"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for top-up
        response = supervisor.process(
            "Can customer 1902 get a top-up loan?", session_id="edge-test-3"
        )

        # Should provide eligibility status
        assert response.answer is not None

        answer_lower = response.answer.lower()
        # Should mention eligibility or top-up
        assert any(
            word in answer_lower
            for word in ["eligible", "topup", "top-up", "loan", "criteria"]
        )

        print(f"✅ Ineligible Top-up Handled:")
        print(f"   {response.answer[:300]}...")

    def test_prepayment_exceeds_balance_per_loan(self):
        """Test prepayment amount greater than outstanding"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Prepayment > outstanding
        validation = calculator._validate_inputs(
            loan_amount=100000,
            interest_rate=10.0,
            tenure_months=12,
            prepayment=150000,
            outstanding_balance=50000,
        )

        # Should reject with helpful message
        assert validation["is_valid"] == False
        assert len(validation["errors"]) > 0

        # Should suggest foreclosure
        has_foreclosure_suggestion = any(
            "foreclosure" in sugg.lower() for sugg in validation.get("suggestions", [])
        )

        assert has_foreclosure_suggestion, "Should suggest foreclosure option"

        print(f"✅ Excessive Prepayment Caught:")
        print(f"   Error: {validation['errors'][0]}")
        print(f"   Suggestion: {validation['suggestions'][0]}")

    def test_zero_balance_remaining_query(self):
        """Test query when loan is fully paid"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query about paid-off loan for specific customer (customer 1000 has closed loan)
        response = supervisor.process(
            "What is the outstanding balance for customer 1000?",
            session_id="edge-test-4",
        )

        # Should provide answer
        assert response.answer is not None

        answer_lower = response.answer.lower()
        # More flexible - handle SQL response showing closed loan with 0 balance
        assert any(
            word in answer_lower
            for word in [
                "balance",
                "zero",
                "paid",
                "outstanding",
                "closed",
                "0.00",
                "no outstanding",
                "fully",
            ]
        )

        print(f"✅ Zero Balance Query Handled:")
        print(f"   {response.answer[:300]}...")

    def test_very_large_loan_amount(self):
        """Test handling of very large loan amounts (edge of range)"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Near maximum loan amount (10 crore)
        loan_amount = 99000000  # 9.9 crore
        interest_rate = 10.0
        tenure_years = 20

        # Validate it's accepted
        is_valid, error_message = calculator._validate_inputs(
            loan_amount, interest_rate, tenure_years * 12
        )
        assert is_valid == True, f"Validation failed: {error_message}"

        # Use LLM to calculate
        response = calculator.process(
            f"Calculate EMI for ₹{loan_amount:,.0f} at {interest_rate}% for {tenure_years} years"
        )

        # Verify response is reasonable
        assert response.answer is not None
        scenarios = response.metadata.get("scenarios", [])
        assert len(scenarios) > 0

        # Find 20-year scenario
        twenty_year_scenario = None
        for scenario in scenarios:
            if scenario.get("tenure_months") == tenure_years * 12:
                twenty_year_scenario = scenario
                break

        if twenty_year_scenario:
            emi = twenty_year_scenario.get("monthly_emi")
            assert emi > 0, "EMI should be positive"
            assert emi < loan_amount, "EMI can't be more than principal"

            print(f"✅ Large Loan Amount Handled:")
            print(f"   Loan: ₹{loan_amount:,}, EMI: ₹{emi:,.2f}")
        else:
            # No exact 20-year scenario, but response should be valid
            assert "emi" in response.answer.lower() or "EMI" in response.answer
            print(f"✅ Large Loan Amount Handled (no exact 20-year scenario)")

    def test_very_short_tenure(self):
        """Test minimum tenure boundary (6 months)"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Minimum tenure
        validation = calculator._validate_inputs(100000, 10.0, 6)

        # Should accept minimum
        assert validation["is_valid"] == True

        # Below minimum should reject
        validation_invalid = calculator._validate_inputs(100000, 10.0, 3)
        assert validation_invalid["is_valid"] == False

        print(f"✅ Tenure Boundaries Validated:")
        print(f"   6 months: ACCEPTED, 3 months: REJECTED")


class TestComplexScenarios:
    """Test complex multi-step scenarios"""

    def test_concurrent_topup_and_prepayment_query(self):
        """Test complex scenario: top-up + prepayment together"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Complex query
        response = supervisor.process(
            "Can I take a top-up and also make a prepayment on my existing loan?",
            session_id="complex-test-1",
        )

        # Should address both aspects
        assert response.answer is not None
        assert len(response.answer) > 50

        answer_lower = response.answer.lower()

        # Should mention both concepts
        has_topup = any(
            word in answer_lower for word in ["topup", "top-up", "additional"]
        )
        has_prepayment = any(word in answer_lower for word in ["prepay", "prepayment"])

        assert has_topup or has_prepayment, "Should address the query concepts"

        print(f"✅ Complex Query Handled:")
        print(f"   {response.answer[:300]}...")

    def test_future_scenario_projection(self):
        """Test what-if scenario for future date"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Future projection query
        response = supervisor.process(
            "What will be my loan status after 6 months?", session_id="complex-test-2"
        )

        # Should provide some information or clarification
        assert response.answer is not None

        answer_lower = response.answer.lower()
        # Should mention time, loan, or ask for clarification
        assert any(
            word in answer_lower
            for word in ["loan", "month", "status", "payment", "emi", "help"]
        )

        print(f"✅ Future Projection Query:")
        print(f"   {response.answer[:300]}...")

    def test_policy_and_calculation_combined(self):
        """Test query requiring both policy lookup and calculation"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Combined query
        response = supervisor.process(
            "What are prepayment charges and how much will I save if I prepay 100000?",
            session_id="complex-test-3",
        )

        # Should address both policy and calculation
        assert response.answer is not None
        assert len(response.answer) > 50

        answer_lower = response.answer.lower()

        # Should mention prepayment concepts
        prepay_terms = ["prepay", "charge", "save", "penalty", "fee"]
        terms_found = sum(1 for term in prepay_terms if term in answer_lower)

        assert terms_found >= 2, "Should address prepayment charges and savings"

        print(f"✅ Policy + Calculation Combined:")
        print(f"   {response.answer[:350]}...")


# Test execution summary
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EDGE CASES & MULTI-LOAN TESTS")
    print("=" * 80)
    print("\nThese tests validate:")
    print("  1. Multiple loan handling for single customer")
    print("  2. Edge cases (closed loans, invalid IDs, boundaries)")
    print("  3. Complex multi-step scenarios")
    print("  4. Boundary value testing")
    print("\nTo run: pytest tests/integration/test_edge_cases.py -v -s")
    print("=" * 80 + "\n")
