-- Casefile domain schema: what a vehicle repair normally costs.
--
-- THIS TABLE IS A HYPOTHETICAL REFERENCE, not an industry price list.
-- casefile.data.repair_cost_reference derives it from the loaded claims: for
-- each incident type and severity, the range that holds the middle 90% of the
-- vehicle_claim amounts. A claim outside its range is unusual for this
-- portfolio, which is a reason to look closer and nothing more.

CREATE TABLE IF NOT EXISTS repair_cost_reference (
    incident_type     ENUM(
        'Single Vehicle Collision',
        'Multi-vehicle Collision',
        'Vehicle Theft',
        'Parked Car'
    )                                NOT NULL,
    incident_severity ENUM(
        'Trivial Damage',
        'Minor Damage',
        'Major Damage',
        'Total Loss'
    )                                NOT NULL,
    low_amount        DECIMAL(12, 2) NOT NULL
        COMMENT '5th percentile of vehicle_claim, USD',
    typical_amount    DECIMAL(12, 2) NOT NULL
        COMMENT 'Median of vehicle_claim, USD',
    high_amount       DECIMAL(12, 2) NOT NULL
        COMMENT '95th percentile of vehicle_claim, USD',
    sample_size       INT UNSIGNED   NOT NULL
        COMMENT 'Number of claims the range was derived from',
    PRIMARY KEY (incident_type, incident_severity),
    CONSTRAINT chk_repair_cost_reference_order
        CHECK (
            low_amount >= 0
            AND low_amount <= typical_amount
            AND typical_amount <= high_amount
        ),
    CONSTRAINT chk_repair_cost_reference_sample
        CHECK (sample_size > 0)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;
