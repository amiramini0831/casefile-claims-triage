-- Casefile domain schema: the human gate.
--
-- Nothing is written back to the claims system unless a person approved it.
-- That rule is enforced here, by the database, and not only by application
-- code:
--
--   recommendations  what the system proposes for a claim
--   approvals        what a person decided about one recommendation
--   write_backs      the final decision recorded against the claim
--
-- A write-back row must point at an approval for the same claim whose
-- decision is 'approved'; the composite foreign key below makes any other
-- row impossible to insert. Unique keys make each step happen at most once:
-- one recommendation per run, one decision per recommendation, one write-back
-- per approval and one per claim.
--
-- The accounts are separated to match (see scripts/setup_database.py): the
-- system's own account can insert recommendations and write-backs but not
-- approvals, and the agents' account can touch none of these tables.

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_id            VARCHAR(64)     NOT NULL
        COMMENT 'The run that produced this recommendation',
    case_id           VARCHAR(20)     NOT NULL,
    claim_number      VARCHAR(20)     NOT NULL
        COMMENT 'As read from the documents',
    policy_number     INT UNSIGNED    NOT NULL
        COMMENT 'As read from the documents; may match no policy',
    outcome           ENUM(
        'recommend_pay',
        'recommend_deny',
        'refer_to_adjuster'
    )                                 NOT NULL,
    reason            VARCHAR(40)     NOT NULL,
    payable_amount    DECIMAL(12, 2)  NOT NULL DEFAULT 0
        COMMENT 'USD; 0 unless the outcome is recommend_pay',
    rationale         TEXT            NOT NULL,
    created_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (recommendation_id),
    UNIQUE KEY uq_recommendations_run (run_id),
    -- Target of the foreign key from approvals.
    UNIQUE KEY uq_recommendations_id_claim (recommendation_id, claim_number),
    CONSTRAINT fk_recommendations_claim
        FOREIGN KEY (claim_number) REFERENCES claims (claim_id),
    CONSTRAINT chk_recommendations_amount
        CHECK ((outcome = 'recommend_pay') = (payable_amount > 0))
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS approvals (
    approval_id       BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    recommendation_id BIGINT UNSIGNED NOT NULL,
    claim_number      VARCHAR(20)     NOT NULL,
    decision          ENUM('approved', 'rejected') NOT NULL,
    approver          VARCHAR(64)     NOT NULL
        COMMENT 'The person who decided',
    comment           VARCHAR(500)    NULL,
    decided_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (approval_id),
    UNIQUE KEY uq_approvals_recommendation (recommendation_id),
    -- Target of the foreign key from write_backs.
    UNIQUE KEY uq_approvals_id_claim_decision (approval_id, claim_number, decision),
    CONSTRAINT fk_approvals_recommendation
        FOREIGN KEY (recommendation_id, claim_number)
        REFERENCES recommendations (recommendation_id, claim_number),
    CONSTRAINT chk_approvals_approver
        CHECK (CHAR_LENGTH(TRIM(approver)) > 0)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS write_backs (
    write_back_id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    approval_id       BIGINT UNSIGNED NOT NULL,
    claim_number      VARCHAR(20)     NOT NULL,
    -- Always 'approved'. It exists so the foreign key below can only match an
    -- approval that was granted, never one that was rejected.
    approval_decision ENUM('approved', 'rejected') NOT NULL DEFAULT 'approved',
    written_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (write_back_id),
    UNIQUE KEY uq_write_backs_approval (approval_id),
    UNIQUE KEY uq_write_backs_claim (claim_number),
    CONSTRAINT fk_write_backs_approval
        FOREIGN KEY (approval_id, claim_number, approval_decision)
        REFERENCES approvals (approval_id, claim_number, decision),
    CONSTRAINT chk_write_backs_approved
        CHECK (approval_decision = 'approved')
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci;
