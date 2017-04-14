DROP TABLE IF EXISTS public.train;
CREATE TABLE public.train
(bidder_id text, payment_account text, address text, outcome double precision);
COPY public.train FROM '/Users/wbarbour1/Downloads/train.csv' DELIMITER ',' CSV HEADER;