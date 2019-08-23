CREATE TYPE emotion AS ENUM ('anger', 'happiness', 'sadness', 'neutral');

CREATE TABLE labels (
  id SERIAL UNIQUE,
  filepath text PRIMARY KEY,
  gender boolean,
  acted boolean,
  emotion emotion,
  arousal smallint,
  valence smallint,
  speaker_number integer, 
  corpus text,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE frames (
  id SERIAL PRIMARY KEY,
  instant smallint,
	f0 DOUBLE PRECISION,
  zcr DOUBLE PRECISION,
  energy DOUBLE PRECISION,
  mfcc1 DOUBLE PRECISION,
  mfcc2 DOUBLE PRECISION,
  mfcc3 DOUBLE PRECISION,
  mfcc4 DOUBLE PRECISION,
  mfcc5 DOUBLE PRECISION,
  mfcc6 DOUBLE PRECISION,
  mfcc7 DOUBLE PRECISION,
  mfcc8 DOUBLE PRECISION,
  mfcc9 DOUBLE PRECISION,
  mfcc10 DOUBLE PRECISION,
  mfcc11 DOUBLE PRECISION,
  mfcc12 DOUBLE PRECISION,
  delta_mfcc1 DOUBLE PRECISION,
  delta_mfcc2 DOUBLE PRECISION,
  delta_mfcc3 DOUBLE PRECISION,
  delta_mfcc4 DOUBLE PRECISION,
  delta_mfcc5 DOUBLE PRECISION,
  delta_mfcc6 DOUBLE PRECISION,
  delta_mfcc7 DOUBLE PRECISION,
  delta_mfcc8 DOUBLE PRECISION,
  delta_mfcc9 DOUBLE PRECISION,
  delta_mfcc10 DOUBLE PRECISION,
  delta_mfcc11 DOUBLE PRECISION,
  delta_mfcc12 DOUBLE PRECISION,
  label_ integer REFERENCES labels(id),
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT unique_instant_label UNIQUE (instant, label_));

/* Insert trigger for adding created_at for newly added records */
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$
LANGUAGE PLPGSQL;

/* Insert trigger for adding created_at for updated records */
CREATE TRIGGER set_timestamp
BEFORE UPDATE ON frames
FOR EACH ROW
EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON labels
FOR EACH ROW
EXECUTE PROCEDURE trigger_set_timestamp();

