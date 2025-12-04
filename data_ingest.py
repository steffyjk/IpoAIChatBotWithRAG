import pandas as pd
from pathlib import Path

CSV_PATH = Path("ipo.csv")  # adjust if your filename differs
OUT_JSON = Path("ipo_docs.csv")

def row_to_text(row):
    # Create readable text for each IPO row. Keep all important fields.
    parts = []
    parts.append(f"IPO Name: {row.get('IPO_Name','')}")
    parts.append(f"Date Opened: {row.get('Date','')}")
    parts.append(f"Issue Size (crores): {row.get('Issue_Size (crores)','')}")
    parts.append(f"Offer Price: {row.get('Offer Price','')}")
    parts.append(f"List Price: {row.get('List Price','')}")
    parts.append(f"Listing Gain: {row.get('Listing Gain','')}")
    parts.append(f"QIB subscription: {row.get('QIB','')}")
    parts.append(f"HNI subscription: {row.get('HNI','')}")
    parts.append(f"RII subscription: {row.get('RII','')}")
    parts.append(f"CMP (BSE): {row.get('CMP (BSE)','')}")
    parts.append(f"Current Gains: {row.get('Current Gains','')}")
    # Optionally add note/computed fields
    return "\n".join(parts)

def ingest():
    df = pd.read_csv(CSV_PATH)
    df.fillna("", inplace=True)
    docs = []
    for _, r in df.iterrows():
        docs.append({
            "ipo_name": r.get("IPO_Name",""),
            "date": str(r.get("Date","")),
            "text": row_to_text(r)
        })
    out_df = pd.DataFrame(docs)
    out_df.to_csv(OUT_JSON, index=False)
    print(f"Saved {len(docs)} docs to {OUT_JSON}")

if __name__ == "__main__":
    ingest()
