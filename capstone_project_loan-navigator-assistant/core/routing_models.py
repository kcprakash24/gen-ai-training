"""
Pydantic models for structured outputs across all agents.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any, List
from enum import Enum


class AgentType(str, Enum):
    """Valid agent types for routing"""

    SQL_AGENT = "SQL_AGENT"
    WHAT_IF_CALCULATOR = "WHAT_IF_CALCULATOR"
    POLICY_GURU = "POLICY_GURU"


class RoutingDecision(BaseModel):
    """Pydantic model for comprehensive routing decisions"""

    agent: AgentType = Field(..., description="The agent to route the query to")
    reasoning: str = Field(
        ..., description="Brief explanation of why this agent was chosen"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    needs_customer_data: bool = Field(
        ..., description="Whether this query needs customer-specific data"
    )


class SQLQueryResult(BaseModel):
    """Structured output for SQL Agent queries"""

    query: str = Field(..., description="The SQL query generated")
    explanation: str = Field(..., description="Explanation of what the query does")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the query"
    )
    estimated_rows: Optional[int] = Field(
        None, description="Estimated number of rows returned"
    )

    @validator("query")
    def validate_query(cls, v):
        if not v.strip().upper().startswith(("SELECT", "WITH")):
            raise ValueError("Query must start with SELECT or WITH")
        return v.strip()


class EMICalculation(BaseModel):
    """Structured output for What-If Calculator"""

    loan_amount: float = Field(..., gt=0, description="Principal loan amount")
    interest_rate: float = Field(
        ..., ge=0, le=100, description="Annual interest rate percentage"
    )
    tenure_months: int = Field(..., gt=0, le=600, description="Loan tenure in months")
    monthly_emi: float = Field(..., gt=0, description="Calculated monthly EMI")
    total_interest: float = Field(..., ge=0, description="Total interest payable")
    total_amount: float = Field(..., gt=0, description="Total amount payable")


class EMIScenarios(BaseModel):
    """Multiple EMI scenarios for different tenures"""

    scenarios: List[EMICalculation] = Field(
        ..., description="List of EMI calculations for different tenures"
    )
    recommendation: str = Field(
        ..., description="Brief recommendation based on calculations"
    )
    summary: str = Field(..., description="Summary of all scenarios")

    @validator("scenarios")
    def validate_scenarios(cls, v):
        if len(v) < 1:
            raise ValueError("At least one scenario is required")
        return v


class PolicyResponse(BaseModel):
    """Simplified policy response model"""

    answer: str = Field(..., description="The policy answer or guidance")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in the answer"
    )

    @validator("answer")
    def validate_answer(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Answer must be at least 10 characters")
        return v.strip()


class LoanInputValidation(BaseModel):
    """Pydantic model for validating loan calculation inputs"""

    loan_amount: float = Field(
        ...,
        gt=0,
        ge=1000,
        le=100000000,
        description="Loan amount between ₹1,000 and ₹10 crore",
    )
    interest_rate: float = Field(
        ...,
        gt=0,
        ge=0.1,
        le=50,
        description="Annual interest rate between 0.1% and 50%",
    )
    tenure_months: Optional[int] = Field(
        None,
        gt=0,
        ge=6,
        le=360,
        description="Loan tenure between 6 months and 30 years",
    )
    prepayment: Optional[float] = Field(
        None, ge=0, description="Prepayment amount (must be non-negative)"
    )
    outstanding_balance: Optional[float] = Field(
        None, ge=0, description="Outstanding balance (must be non-negative)"
    )

    @validator("prepayment")
    def validate_prepayment(cls, v, values):
        """Ensure prepayment doesn't exceed loan amount or outstanding balance"""
        if v is None:
            return v

        # Check against outstanding balance first
        if (
            "outstanding_balance" in values
            and values["outstanding_balance"] is not None
        ):
            if v > values["outstanding_balance"]:
                raise ValueError(
                    f"Prepayment (₹{v:,.0f}) exceeds outstanding balance (₹{values['outstanding_balance']:,.0f})"
                )
        # Otherwise check against loan amount
        elif "loan_amount" in values:
            if v > values["loan_amount"]:
                raise ValueError(
                    f"Prepayment (₹{v:,.0f}) exceeds loan amount (₹{values['loan_amount']:,.0f})"
                )

        return v
