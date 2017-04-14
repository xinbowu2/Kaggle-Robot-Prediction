DROP TABLE IF EXISTS public.test;
CREATE TABLE public.test
(bidder_id text, payment_account text, address text);
COPY public.test FROM '/Users/wbarbour1/Downloads/test.csv' DELIMITER ',' CSV HEADER;