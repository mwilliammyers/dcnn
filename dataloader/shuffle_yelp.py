import csv
import random

n = 100000

with open("/mnt/pccfs/not_backed_up/data/yelp/yelp_review_full.csv") as f:
    reader = csv.DictReader(f)
    sample = random.sample(list(reader), n)

with open(f"/mnt/pccfs/not_backed_up/data/yelp/yelp_review_{n}.csv", "w") as f:
    fieldnames = ["review_id", "user_id", "business_id", "stars", "date", "text", "useful", "funny", "cool"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sample:
        writer.writerow(r)
