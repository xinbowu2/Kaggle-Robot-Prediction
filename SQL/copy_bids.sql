DROP TABLE IF EXISTS public.bids;
CREATE TABLE public.bids
(bid_id integer, bidder_id text, auction text, merchandise text, device text, "time" bigint, country char(2), ip text, url text);
COPY public.bids FROM '/Users/wbarbour1/Downloads/bids.csv' DELIMITER ',' CSV HEADER;