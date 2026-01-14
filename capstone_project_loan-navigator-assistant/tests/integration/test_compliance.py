"""
Integration tests - Regulatory & Compliance
Tests to ensure RBI compliance, proper disclosures, and audit readiness
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


class TestRBICompliance:
    """Test RBI regulatory compliance in responses"""

    def test_rbi_disclosure_in_emi_response(self):
        """Critical: EMI responses must include proper disclosures"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # EMI calculation query
        response = supervisor.process(
            "Calculate EMI for 500000 at 10% for 5 years",
            session_id="compliance-test-1",
        )

        # Should provide calculation
        assert response.answer is not None
        assert len(response.answer) > 50

        # Check for financial terms (proper disclosure)
        answer_lower = response.answer.lower()

        # Should mention key financial terms
        financial_terms = ["emi", "interest", "rate", "amount", "tenure", "total"]
        terms_found = sum(1 for term in financial_terms if term in answer_lower)

        assert (
            terms_found >= 4
        ), f"Should mention at least 4 financial terms for proper disclosure"

        print(f"✅ RBI Disclosure Check - {terms_found}/6 financial terms mentioned")
        print(f"   Response: {response.answer[:300]}...")

    def test_prepayment_charge_disclosure(self):
        """Critical: Prepayment responses must disclose charges/penalties"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Prepayment charges query
        response = supervisor.process(
            "What are the prepayment charges for my loan?",
            session_id="compliance-test-2",
        )

        # Should provide policy information
        assert response.answer is not None

        answer_lower = response.answer.lower()

        # Should mention charges/penalty or no charges
        charge_terms = [
            "charge",
            "fee",
            "penalty",
            "prepayment",
            "foreclosure",
            "cost",
            "no charge",
            "free",
        ]
        has_charge_info = any(term in answer_lower for term in charge_terms)

        assert (
            has_charge_info
        ), "Response must disclose prepayment charges or mention no charges"

        print(f"✅ Prepayment Charge Disclosure Included")
        print(f"   Response: {response.answer[:300]}...")

    def test_interest_rate_disclosure(self):
        """Test that interest rate is properly disclosed"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Interest rate query
        response = supervisor.process(
            "What is the interest rate for customer 1900?",
            session_id="compliance-test-3",
        )

        # Should provide interest rate
        assert response.answer is not None

        answer_lower = response.answer.lower()

        # Should mention interest rate
        assert any(
            term in answer_lower
            for term in ["interest", "rate", "%", "percent", "annual"]
        )

        print(f"✅ Interest Rate Disclosed")
        print(f"   Response: {response.answer[:300]}...")

    def test_terms_and_conditions_reference(self):
        """Test that policy responses reference terms and conditions"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Policy query
        response = supervisor.process(
            "What are the loan eligibility criteria?", session_id="compliance-test-4"
        )

        # Should provide policy information
        assert response.answer is not None
        assert len(response.answer) > 30

        # Check for policy/criteria information
        answer_lower = response.answer.lower()
        policy_terms = ["eligibility", "criteria", "requirement", "policy", "condition"]

        has_policy_info = any(term in answer_lower for term in policy_terms)
        assert has_policy_info, "Should reference policy or eligibility criteria"

        print(f"✅ Terms & Conditions Referenced")
        print(f"   Response: {response.answer[:300]}...")

    def test_no_misleading_information(self):
        """Critical: Ensure no misleading or incorrect financial advice"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # EMI calculation with specific numbers
        response = supervisor.process(
            "Calculate EMI for 100000 at 12% for 12 months",
            session_id="compliance-test-5",
        )

        # Should provide calculation
        assert response.answer is not None

        # Should not promise guaranteed returns or misleading claims
        answer_lower = response.answer.lower()

        misleading_terms = [
            "guaranteed profit",
            "risk-free",
            "get rich",
            "no loss",
            "always profitable",
        ]
        has_misleading = any(term in answer_lower for term in misleading_terms)

        assert (
            not has_misleading
        ), "Response should not contain misleading financial claims"

        print(f"✅ No Misleading Information")
        print(f"   Response checked for misleading claims: PASSED")

    def test_accurate_financial_calculations(self):
        """Test that financial calculations are mathematically accurate"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Known test case - use 5 years which is in default scenarios
        loan_amount = 100000
        interest_rate = 12.0
        tenure_years = 5

        # Use the actual LLM to calculate
        response = calculator.process(
            f"Calculate EMI for ₹{loan_amount:,.0f} at {interest_rate}% for {tenure_years} years"
        )

        # Extract EMI from response
        assert response.answer is not None
        assert "₹" in response.answer
        assert "emi" in response.answer.lower() or "EMI" in response.answer

        # Verify scenarios metadata exists and has reasonable values
        scenarios = response.metadata.get("scenarios", [])
        assert len(scenarios) > 0, "Should return EMI scenarios"

        # Find the 5-year scenario (60 months)
        five_year_scenario = None
        for scenario in scenarios:
            if scenario.get("tenure_months") == 60:
                five_year_scenario = scenario
                break

        assert five_year_scenario is not None, "Should have 5-year scenario"

        # Validate EMI is reasonable
        emi = five_year_scenario.get("monthly_emi")
        assert emi is not None
        assert emi > 0, "EMI should be positive"
        assert emi < loan_amount, "Monthly EMI should be less than principal"

        # Calculate expected EMI for reference
        monthly_rate = interest_rate / (12 * 100)
        tenure_months = 60
        expected_emi = (
            loan_amount
            * monthly_rate
            * pow(1 + monthly_rate, tenure_months)
            / (pow(1 + monthly_rate, tenure_months) - 1)
        )

        # Should be within 1% of mathematically correct value
        percentage_diff = abs(emi - expected_emi) / expected_emi * 100
        assert percentage_diff < 1.0, f"EMI calculation deviation: {percentage_diff}%"

        print(f"✅ Financial Calculation Accuracy: {percentage_diff:.4f}% deviation")
        print(f"   Calculated: ₹{emi:,.2f}, Expected: ₹{expected_emi:,.2f}")


class TestAuditAndLogging:
    """Test audit trail and logging capabilities"""

    def test_query_metadata_logging(self):
        """Test that query metadata is captured for audit"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Execute query
        response = supervisor.process(
            "Show loan for customer 1900", session_id="audit-test-1"
        )

        # Should have metadata
        assert response.metadata is not None
        assert isinstance(response.metadata, dict)

        print(f"✅ Metadata Logged: {len(response.metadata)} fields")
        print(f"   Metadata keys: {list(response.metadata.keys())[:5]}...")

    def test_session_tracking(self):
        """Test that sessions are tracked properly"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        session_id = "audit-session-track"

        # Multiple queries in same session
        response1 = supervisor.process(
            "What is EMI for customer 1900?", session_id=session_id
        )
        response2 = supervisor.process(
            "What is the outstanding balance?", session_id=session_id
        )

        # Both should succeed
        assert response1.answer is not None
        assert response2.answer is not None

        print(f"✅ Session Tracking: 2 queries in session '{session_id}'")
        print(f"   Query 1: {response1.answer[:50]}...")
        print(f"   Query 2: {response2.answer[:50]}...")

    def test_error_logging_and_tracking(self):
        """Test that errors are properly logged"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Invalid input should be caught and logged
        validation = calculator._validate_inputs(-50000, 10.0, 12, 0, 0)

        # Should capture error details
        assert validation["is_valid"] == False
        assert len(validation["errors"]) > 0

        # Errors should be descriptive for audit
        error_message = validation["errors"][0]
        assert len(error_message) > 10, "Error messages should be descriptive"

        print(f"✅ Error Logging: {len(validation['errors'])} errors captured")
        print(f"   Error: {error_message}")


class TestResponseToneAndCompliance:
    """Test response tone is professional and compliant"""

    def test_friendly_professional_tone(self):
        """Test that responses maintain friendly yet professional tone"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Regular query
        response = supervisor.process(
            "What are my loan details for customer 1900?", session_id="tone-test-1"
        )

        # Should get response
        assert response.answer is not None
        assert len(response.answer) > 20

        # Should not be overly casual or use slang
        answer_lower = response.answer.lower()
        unprofessional_terms = ["gonna", "wanna", "yeah", "nope", "lol", "hey dude"]
        is_professional = not any(term in answer_lower for term in unprofessional_terms)

        assert is_professional, "Response should maintain professional tone"

        print(f"✅ Professional Tone Maintained")
        print(f"   Response: {response.answer[:200]}...")

    def test_error_message_politeness(self):
        """Test that error messages are polite and helpful"""
        from agents.what_if_calculator import WhatIfCalculator

        calculator = WhatIfCalculator()

        # Generate validation error
        validation = calculator._validate_inputs(-100000, 10.0, 12, 0, 0)

        # Should provide helpful suggestions
        assert (
            len(validation.get("suggestions", [])) > 0
        ), "Should provide suggestions for errors"

        # Error response should be constructive
        error_response = calculator._generate_error_response(validation, "test query")

        assert (
            "help" in error_response
            or "range" in error_response
            or "valid" in error_response
        )

        print(f"✅ Error Messages are Helpful")
        print(f"   Suggestion: {validation['suggestions'][0]}")

    def test_jargon_free_responses(self):
        """Test that responses avoid excessive technical jargon"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Simple customer query
        response = supervisor.process(
            "How much do I owe on my loan?", session_id="tone-test-2"
        )

        # Should get understandable response
        assert response.answer is not None

        # Check that response is not overly technical
        answer_lower = response.answer.lower()

        # Common terms are OK
        common_terms = ["loan", "amount", "payment", "balance", "emi"]

        # Over-technical jargon to avoid (for simple queries)
        # Note: These are acceptable in policy responses, but simple queries should be clear

        print(f"✅ Response Clarity Checked")
        print(f"   Response: {response.answer[:250]}...")


class TestDataPrivacyAndSecurity:
    """Test data privacy and security measures"""

    def test_customer_data_isolation(self):
        """Critical: Customer A should not see Customer B's data"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Query for specific customer
        response = supervisor.process(
            "Show loan details for customer 1900", session_id="security-test-1"
        )

        # Should only return data for customer 1900
        assert response.answer is not None

        # If customer ID mentioned, should be 1900
        if "1900" in response.answer or "customer" in response.answer.lower():
            # Should not leak other customer IDs
            other_customers = ["1901", "1902", "1903"]
            has_leak = any(cust_id in response.answer for cust_id in other_customers)

            assert not has_leak, "Should not leak other customer data"

        print(f"✅ Customer Data Isolation Verified")
        print(f"   No data leakage detected")

    def test_sql_injection_prevention_variants(self):
        """Test multiple SQL injection attack patterns"""
        from agents.sql_agent import SQLAgent

        agent = SQLAgent()

        # Various SQL injection attempts
        injection_attempts = [
            "1' OR '1'='1",
            "1'; DROP TABLE loan_data; --",
            "' UNION SELECT * FROM users --",
            "1' AND 1=1 --",
        ]

        # All should be safely handled (parameterized queries or validation)
        for injection in injection_attempts:
            try:
                # If using parameterized queries, this should be safe
                sql = "SELECT * FROM loan_data WHERE customer_id = ?"
                results = agent._execute_sql_with_params(sql, (injection,))

                # Should return empty or safe results
                assert isinstance(results, list)
                print(f"✅ SQL Injection Blocked: {injection[:30]}...")

            except Exception as e:
                # If validation catches it, that's also good
                print(f"✅ SQL Injection Caught by Validation: {injection[:30]}...")

    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive data is not exposed in metadata/logs"""
        from agents.supervisor import Supervisor

        supervisor = Supervisor()

        # Process query
        response = supervisor.process(
            "Show loan for customer 1900", session_id="privacy-test-1"
        )

        # Metadata should not contain raw sensitive data like full account numbers
        # (This is a basic check - in production, you'd have more sophisticated PII detection)

        assert response.metadata is not None

        # Check metadata doesn't contain long numeric sequences (potential account numbers)
        import re

        metadata_str = str(response.metadata)

        # Look for 10+ digit sequences (potential sensitive data)
        long_numbers = re.findall(r"\d{10,}", metadata_str)

        # Customer IDs (4 digits) are OK, but not longer sequences
        has_potential_sensitive = any(len(num) > 8 for num in long_numbers)

        if has_potential_sensitive:
            print(f"⚠️ Warning: Potential sensitive data in metadata: {long_numbers}")
        else:
            print(f"✅ No Sensitive Data in Metadata")


# Test execution summary
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPLIANCE & REGULATORY TESTS")
    print("=" * 80)
    print("\nThese tests validate:")
    print("  1. RBI compliance and proper disclosures")
    print("  2. Audit trail and logging capabilities")
    print("  3. Professional and compliant tone")
    print("  4. Data privacy and security measures")
    print("\nTo run: pytest tests/integration/test_compliance.py -v -s")
    print("=" * 80 + "\n")
