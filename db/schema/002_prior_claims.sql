-- Casefile domain schema: earlier claims of each policy.
--
-- EVERY ROW IN THIS TABLE IS SYNTHETIC. Neither source dataset contains a
-- claim history, so casefile.data.prior_claims generates one with a fixed
-- seed. Only the number of earlier claims per policy follows real data (the
-- PastNumberOfClaims column of fraud_oracle.csv); dates are drawn at random
-- and the content of each row is copied from another policy's claim.
--
-- History belongs to the policy, not to the insured vehicle, so there are no
-- vehicle columns here.

CREATE TABLE IF NOT EXISTS prior_claims (
    prior_claim_id     VARCHAR(20)    NOT NULL
        COMMENT 'Synthetic, e.g. PCL-00001',
    policy_number      INT UNSIGNED   NOT NULL,
    incident_date      DATE           NOT NULL,
    incident_type      ENUM(
        'Single Vehicle Collision',
        'Multi-vehicle Collision',
        'Vehicle Theft',
        'Parked Car'
    )                                 NOT NULL,
    incident_severity  ENUM(
        'Trivial Damage',
        'Minor Damage',
        'Major Damage',
        'Total Loss'
    )                                 NOT NULL,
    total_claim_amount DECIMAL(12, 2) NOT NULL COMMENT 'USD',
    injury_claim       DECIMAL(12, 2) NOT NULL COMMENT 'USD',
    property_claim     DECIMAL(12, 2) NOT NULL COMMENT 'USD',
    vehicle_claim      DECIMAL(12, 2) NOT NULL COMMENT 'USD',
    PRIMARY KEY (prior_claim_id),
    -- At most one earlier claim per policy per day.
    UNIQUE KEY uq_prior_claims_policy_date (policy_number, incident_date),
    -- Reloading the policies removes their generated history with them.
    CONSTRAINT fk_prior_claims_policy
        FOREIGN KEY (policy_number) REFERENCES policies (policy_number)
        ON DELETE CASCADE
        ON UPDATE RESTRICT,
    CONSTRAINT chk_prior_claims_amounts_non_negative
        CHECK (
            injury_claim >= 0
            AND property_claim >= 0
            AND vehicle_claim >= 0
        ),
    CONSTRAINT chk_prior_claims_total_is_sum
        CHECK (
            total_claim_amount = injury_claim + property_claim + vehicle_claim
        )
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;
