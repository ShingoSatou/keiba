BEGIN;

ALTER TABLE core.runner ADD COLUMN IF NOT EXISTS trainer_code_raw TEXT;
ALTER TABLE core.runner ADD COLUMN IF NOT EXISTS trainer_name_abbr TEXT;
ALTER TABLE core.runner ADD COLUMN IF NOT EXISTS jockey_code_raw TEXT;
ALTER TABLE core.runner ADD COLUMN IF NOT EXISTS jockey_name_abbr TEXT;

COMMIT;
