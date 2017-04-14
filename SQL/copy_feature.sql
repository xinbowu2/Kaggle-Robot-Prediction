﻿DROP TABLE IF EXISTS public.features
CREATE TABLE public.features
("n" integer,	"bidder_id" text,	"outcome" double precision,	"payment_account_prefix_same_as_address_prefix" boolean,	"address_10892096f6d88d7449a249e831bfc8d8" double precision,	"address_111d4514cb2db9cc0aaf1ae65b39e70a" double precision,	"address_18efe9aa2a264f71020ed46242cab513" double precision,	"address_2367520c8d94e70f47f4e28e72004ef9" double precision,	"address_26108bd7fdfd14d2b7bf24aec40d86b2" double precision,	"address_274a03538f87ee7ec8b091da1c4fa8fc" double precision,	"address_2a96c3ce94b3be921e0296097b88b56a" double precision,	"address_2cf1e0523f43138fd65eebb2c8cb4b73" double precision,	"address_3a0b8057e5a900d872232c62a7f4c120" double precision,	"address_3a7e6a32b24aeab0688e91a41f3188e2" double precision,	"address_3ee4b535441c1b35d3f9aa3f7ed5ab88" double precision,	"address_4544d31a6d8a15cdee64e7dafbc63e6d" double precision,	"address_4a28d0cf109896236a89b2a1ab33b50c" double precision,	"address_4d90f2e709f1fc0810e5aef472dd3935" double precision,	"address_5c9de1da50cc32a29ffd596ae24cd2be" double precision,	"address_63ce78587e427547c60b44e54d51a4c9" double precision,	"address_7578f951008bd0b64528bf81b8578d5d" double precision,	"address_794abfcfc9a51569c6415a61a319b352" double precision,	"address_8c7fc2ee1e693a7f6e0c986c7375e0fc" double precision,	"address_8db1dcb1ae23a9b13d2a7f36bc991f00" double precision,	"address_91ea8f05b2be1f7fe36e86e20ea35105" double precision,	"address_937e24e5329b1758d1180c3b83041b1b" double precision,	"address_a3d2de7675556553a5f08e4c88d2c228" double precision,	"address_a71773a409dd8baba89ac1c7b4e4523a" double precision,	"address_aa497a035e67fdde98e547c22102646d" double precision,	"address_b16a6b39a27e215e5d571472485ef08e" double precision,	"address_b1abeeedc3163ca5d0df0eecbe6923d0" double precision,	"address_b312117de5aeb80a49c753f562c4b2d3" double precision,	"address_b650404e1ab5d177020221277c3e9306" double precision,	"address_b9b03d5a127eb07aeb9163cdcf524e13" double precision,	"address_c00ae99c1ef4854bd838c9648ea6eb51" double precision,	"address_c047e4d26a46f9fb919883da4ee2adef" double precision,	"address_c0688655078b9fd63d343bd4ae21194e" double precision,	"address_c268ac47bde5354c290a7374b3dd1c41" double precision,	"address_c4b65a7ef32586d7f9a711f867d3f54b" double precision,	"address_c6a1c9967d64e7f99ee545c2a5467960" double precision,	"address_c94cf5c3c5205afe0ef14ce669e01565" double precision,	"address_ca8d4b018cb62966eebb2974f5a83b4f" double precision,	"address_cb8cab3337c5bc1bb96ccfb18d6a62f7" double precision,	"address_ce0b318902ae2a4f5cc4023cea883d00" double precision,	"address_d204ba7ab87c1efb88e93a824efc926a" double precision,	"address_dbd1190dd19462115568e721afba7801" double precision,	"address_e00c0bbaf904d8155da1f1fc85af7434" double precision,	"address_e048f5d50823df461686fde13a25d287" double precision,	"address_e23d9777cddc347de82d839b2e54b22e" double precision,	"address_e350bda3ceb6067cad400ae931c3527a" double precision,	"address_e9ac7750d73a796460f96b096b29725f" double precision,	"address_eb0d036dc33a16ce54296758f96b9d0b" double precision,	"address_efed968be472133f24bd70f9412c156d" double precision,	"address_f3b1e4c7e5f5337ab0ff0d66aee32686" double precision,	"address_fa442b098896fe3f9aca16a6f100e597" double precision,	"address_fb5f20b04f48113f484f73988d44a09a" double precision,	"address_infrequent_address" double precision,	"address_rare_address" double precision,	"payment_account_0875307e1731af94b3b64725ad0deb7d" double precision,	"payment_account_0f244480cd73688fab682a82aa37a587" double precision,	"payment_account_1b53c9a459d6cae0a015df2bc2a6ed7f" double precision,	"payment_account_3e9073fb9219ceb4a1dc9dbb9e1acbe9" double precision,	"payment_account_474f441d0c8f8416be60291872926cd5" double precision,	"payment_account_4a347e6808ca4bd52f2f21a759e1beb2" double precision,	"payment_account_5a8c8865eaf65dc094a6bd9d6e7007cf" double precision,	"payment_account_7df4ebd184668b4257f740b11d4519af" double precision,	"payment_account_a0dc95936282ff8eef7ffa54f295255c" double precision,	"payment_account_a27bbbfe50ced2566f17d94553ec7b6b" double precision,	"payment_account_a3d2de7675556553a5f08e4c88d2c228" double precision,	"payment_account_af6c260faf0b9df48f2155e38b0a29e6" double precision,	"payment_account_b35a1080bb5f0ebb4212508b92c7196c" double precision,	"payment_account_bb97065ca7b313ebbee43f495d6f67a0" double precision,	"payment_account_c9274731b884ef7495041d62d4ec512e" double precision,	"payment_account_c9ef8f9c82a24602bca4f1fc4e69fd61" double precision,	"payment_account_dab1ae58a15b808c63d8dfcf734f5ee3" double precision,	"payment_account_infrequent_account" double precision,	"payment_account_rare_account" double precision,	"bids_per_auction_per_ip_entropy_median" double precision,	"bids_per_auction_per_ip_entropy_mean" double precision,	"ips_per_bidder_per_auction_median" double precision,	"ips_per_bidder_per_auction_mean" double precision,	"only_one_user" double precision,	"ip_only_one_user_counts" double precision,	"on_ip_that_has_a_bot" double precision,	"on_ip_that_has_a_bot_mean" double precision,	"ip_entropy" double precision,	"dt_change_ip_median" double precision,	"dt_same_ip_median" double precision,	"num_first_bid" double precision,	"short" double precision,	"t_until_end_median" double precision,	"t_since_start_median" double precision,	"0_hour72" double precision,	"1_hour72" double precision,	"2_hour72" double precision,	"3_hour72" double precision,	"4_hour72" double precision,	"5_hour72" double precision,	"6_hour72" double precision,	"7_hour72" double precision,	"8_hour72" double precision,	"9_hour72" double precision,	"10_hour72" double precision,	"11_hour72" double precision,	"12_hour72" double precision,	"13_hour72" double precision,	"14_hour72" double precision,	"15_hour72" double precision,	"16_hour72" double precision,	"17_hour72" double precision,	"18_hour72" double precision,	"19_hour72" double precision,	"20_hour72" double precision,	"21_hour72" double precision,	"22_hour72" double precision,	"23_hour72" double precision,	"24_hour72" double precision,	"25_hour72" double precision,	"26_hour72" double precision,	"27_hour72" double precision,	"28_hour72" double precision,	"29_hour72" double precision,	"30_hour72" double precision,	"31_hour72" double precision,	"32_hour72" double precision,	"33_hour72" double precision,	"34_hour72" double precision,	"35_hour72" double precision,	"36_hour72" double precision,	"37_hour72" double precision,	"38_hour72" double precision,	"39_hour72" double precision,	"40_hour72" double precision,	"41_hour72" double precision,	"42_hour72" double precision,	"43_hour72" double precision,	"44_hour72" double precision,	"45_hour72" double precision,	"46_hour72" double precision,	"47_hour72" double precision,	"48_hour72" double precision,	"49_hour72" double precision,	"50_hour72" double precision,	"51_hour72" double precision,	"52_hour72" double precision,	"53_hour72" double precision,	"54_hour72" double precision,	"55_hour72" double precision,	"56_hour72" double precision,	"57_hour72" double precision,	"58_hour72" double precision,	"59_hour72" double precision,	"60_hour72" double precision,	"61_hour72" double precision,	"62_hour72" double precision,	"63_hour72" double precision,	"64_hour72" double precision,	"65_hour72" double precision,	"66_hour72" double precision,	"67_hour72" double precision,	"68_hour72" double precision,	"69_hour72" double precision,	"70_hour72" double precision,	"71_hour72" double precision,	"max_bids_in_hour72" double precision,	"sleep" boolean,	"dt_others_median" double precision,	"f_dt_others_lt_cutoff" double precision,	"dt_self_median" double precision,	"dt_self_min" double precision,	"monday" double precision,	"tuesday" double precision,	"wednesday" double precision,	"balance" double precision,	"s_monday" double precision,	"s_tuesday" double precision,	"s_wednesday" double precision,	"f_monday" double precision,	"f_tuesday" double precision,	"f_wednesday" double precision,	"bids_per_auction_median" double precision,	"bids_per_auction_mean" double precision,	"n_bids" double precision,	"n_urls" double precision,	"n_bids_url" double precision,	"f_urls" double precision,	"13tdx9x3c4du6tr" double precision,	"1tjpqb3mqacsct0" double precision,	"2l309145ivgdy6r" double precision,	"301o49axv6udhkl" double precision,	"3ir1yiophclnlyj" double precision,	"d75il608kel7ote" double precision,	"i695h0p5uj10gmq" double precision,	"pw7hob3wyniyp5j" double precision,	"q5draunzzjr7wc2" double precision,	"szyjr65zi6h3qbz" double precision,	"vasstdc27m7nks3" double precision,	"ya0vpnuq1shnign" double precision,	"url_entropy" double precision,	"countries_per_bidder_per_auction_median" double precision,	"countries_per_bidder_per_auction_mean" double precision,	"countries_per_bidder_per_auction_max" double precision,	"most_common_country" text,	"ad" double precision,	"ae" double precision,	"af" double precision,	"ag" double precision,	"al" double precision,	"am" double precision,	"an" double precision,	"ao" double precision,	"ar" double precision,	"at" double precision,	"au" double precision,	"aw" double precision,	"az" double precision,	"ba" double precision,	"bb" double precision,	"bd" double precision,	"be" double precision,	"bf" double precision,	"bg" double precision,	"bh" double precision,	"bi" double precision,	"bj" double precision,	"bm" double precision,	"bn" double precision,	"bo" double precision,	"br" double precision,	"bs" double precision,	"bt" double precision,	"bw" double precision,	"by" double precision,	"bz" double precision,	"ca" double precision,	"cd" double precision,	"cf" double precision,	"cg" double precision,	"ch" double precision,	"ci" double precision,	"cl" double precision,	"cm" double precision,	"cn" double precision,	"co" double precision,	"cr" double precision,	"cv" double precision,	"cy" double precision,	"cz" double precision,	"de" double precision,	"dj" double precision,	"dk" double precision,	"dm" double precision,	"do" double precision,	"dz" double precision,	"ec" double precision,	"ee" double precision,	"eg" double precision,	"er" double precision,	"es" double precision,	"et" double precision,	"eu" double precision,	"fi" double precision,	"fj" double precision,	"fo" double precision,	"fr" double precision,	"ga" double precision,	"gb" double precision,	"ge" double precision,	"gh" double precision,	"gi" double precision,	"gl" double precision,	"gm" double precision,	"gn" double precision,	"gp" double precision,	"gq" double precision,	"gr" double precision,	"gt" double precision,	"gu" double precision,	"gy" double precision,	"hk" double precision,	"hn" double precision,	"hr" double precision,	"ht" double precision,	"hu" double precision,	"id" double precision,	"ie" double precision,	"il" double precision,	"in" double precision,	"iq" double precision,	"ir" double precision,	"is" double precision,	"it" double precision,	"je" double precision,	"jm" double precision,	"jo" double precision,	"jp" double precision,	"ke" double precision,	"kg" double precision,	"kh" double precision,	"kr" double precision,	"kw" double precision,	"kz" double precision,	"la" double precision,	"lb" double precision,	"li" double precision,	"lk" double precision,	"lr" double precision,	"ls" double precision,	"lt" double precision,	"lu" double precision,	"lv" double precision,	"ly" double precision,	"ma" double precision,	"mc" double precision,	"md" double precision,	"me" double precision,	"mg" double precision,	"mh" double precision,	"mk" double precision,	"ml" double precision,	"mm" double precision,	"mn" double precision,	"mo" double precision,	"mp" double precision,	"mr" double precision,	"mt" double precision,	"mu" double precision,	"mv" double precision,	"mw" double precision,	"mx" double precision,	"my" double precision,	"mz" double precision,	"na" double precision,	"nc" double precision,	"ne" double precision,	"ng" double precision,	"ni" double precision,	"nl" double precision,	"no" double precision,	"np" double precision,	"nz" double precision,	"om" double precision,	"pa" double precision,	"pe" double precision,	"pf" double precision,	"pg" double precision,	"ph" double precision,	"pk" double precision,	"pl" double precision,	"pr" double precision,	"ps" double precision,	"pt" double precision,	"py" double precision,	"qa" double precision,	"re" double precision,	"ro" double precision,	"rs" double precision,	"ru" double precision,	"rw" double precision,	"sa" double precision,	"sb" double precision,	"sc" double precision,	"sd" double precision,	"se" double precision,	"sg" double precision,	"si" double precision,	"sk" double precision,	"sl" double precision,	"sn" double precision,	"so" double precision,	"sr" double precision,	"sv" double precision,	"sy" double precision,	"sz" double precision,	"tc" double precision,	"td" double precision,	"tg" double precision,	"th" double precision,	"tj" double precision,	"tl" double precision,	"tm" double precision,	"tn" double precision,	"tr" double precision,	"tt" double precision,	"tw" double precision,	"tz" double precision,	"ua" double precision,	"ug" double precision,	"uk" double precision,	"us" double precision,	"uy" double precision,	"uz" double precision,	"vc" double precision,	"ve" double precision,	"vi" double precision,	"vn" double precision,	"ws" double precision,	"ye" double precision,	"za" double precision,	"zm" double precision,	"zw" double precision,	"zz" double precision,	"auto_parts" double precision,	"books_and_music" double precision,	"clothing" double precision,	"computers" double precision,	"furniture" double precision,	"home_goods" double precision,	"jewelry" double precision,	"mobile" double precision,	"office_equipment" double precision,	"sporting_goods" double precision);
COPY public.features FROM '/Users/wbarbour1/Downloads/features.csv' DELIMITER ',' CSV HEADER;