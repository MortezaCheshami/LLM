
import pandas as pd
import time
from openai import OpenAI

# Connect to OpenAI
client = OpenAI(api_key="Your API Key")

# File paths
input_path = r"C:\HP - OIT - 2024\Lefel\Task 3\Splited Data\From 1 to 10001.xlsx"
output_path = r"C:\HP - OIT - 2024\Lefel\Task 3\Splited Data\LLM Result Main Data\standardized_facilities_batch_1 to 10001_Final.xlsx"

# Read input data
df = pd.read_excel(input_path)
facilities = df["facility.name.sc1"].tolist()

# Prompt builder for each batch
def build_prompt_batch(names):
    numbered_names = "\n".join([f"{i+1}. {name}" for i, name in enumerate(names)])
    prompt = f"""
You are a facility name standardizer and deduplicator. Your task is to clean, standardize, and unify similar names that refer to the same facility.

Please follow the strict rules below to ensure consistent and accurate results.  
Always return a cleaned version of each input. Do NOT skip or return [unclear].

CLEANING & STANDARDIZATION RULES:

1. Replace only the exact phrase "Gas Plant" with "GP".  
   Do NOT replace "Plant" or "Gas" alone.  
   Example: "Abbiategrosso Gas Plant" → "Abbiategrosso GP"

2. If multiple names refer to the same facility due to typos, location codes, or format variations, unify them into a single standardized version.  
   Example: "The Research Division" and "The Research Division in Japan" → "The Research Division in Japan"

3. Fix obvious typos like "plan" → "plant" when context clearly indicates the correct form.

4. If names differ only by plural form (e.g., “Facility” vs “Facilities”), treat them as the same and use the more common or appropriate form.

5. Preserve acronyms in uppercase if the original name is fully capitalized and short (6 characters or fewer).  
   Example: "KUI", "KSD", "KUCC" → do NOT convert to "Kui", etc.

6. Normalize casing to Title Case, unless the name is a known acronym (fully capitalized and short).  
   Example: "CHONG-QING" → "Chong Qing", "KOM SA" → keep as is

7. For U.S. and Canadian locations with province/state codes (e.g., “Houston, TX”, “Concord, ON”), prefer the version that includes the code.  
   Example: "Houston" + "Houston, TX" → "Houston TX"

8. Remove special characters including: hyphens (-), commas (,), periods (.), slashes (/), parentheses (), quotes ("")  
   Replace all with a single space between words

9. Remove trailing punctuation such as extra dots, commas, or quotation marks.

10. Preserve all numbers and descriptors like: "Phase 1", "East Gate", "Line 3", "05-32", "10-07", etc.

11. Never join words together — always preserve natural word spacing.

12. If two names are identical except for ownership percentages in parentheses (e.g., "Chirano (90%)"), treat them as the same facility.

13. If two names differ only by a typo or character omission (e.g., “Shanghai” vs “Shanghi”), correct the typo and treat them as the same.

14. If a name includes or omits a province/state code (like “TX”, “ON”), always keep the version that includes the code.

15. If a facility name includes location-specific numeric codes (e.g., "10-07", "05-32", "9-6"), treat these as site descriptors, not different facilities.

16. If two facility names differ only by a country name suffix (e.g., "France", "USA", "Germany", "Brazil", etc.), and clearly refer to the same company or entity, then unify them into a single version that includes the country name, as long as no other version of that name exists in a different country.

Examples:
"KOM SA" and "KOM SA France" → "KOM SA France"
"RT Soft" and "RT Soft Russia" → "RT Soft Russia"
"KT A/S" and "KT A/S Denmark" → "KT A/S Denmark"
Apply this even when the country name:
is separated by a comma → "RTSoft, Russia" → "RT Soft Russia"
is separated by a semicolon or parenthesis → "KT A/S (Denmark)" → "KT A/S Denmark"
is appended with just a space → "KT A/S Denmark" is valid
Do not unify if there are different versions of the same company name across multiple countries (e.g., "XYZ Ltd France", "XYZ Ltd USA").

17. If a name includes a known short all-caps acronym or company code like "KIK", "KBR", "IBM", keep the acronym fully capitalized, even if the rest of the name is in Title Case.  
   Example: "KIK Houston" → "KIK Houston"

OUTPUT FORMAT:
Return only the cleaned names in this format:

1. [cleaned name]  
2. [cleaned name]  
...

The number of output names must exactly match the number of input names.  
Do NOT skip, remove, or merge rows. Simply standardize and unify the text content.

Here are the facility names:
{numbered_names}
""".strip()
    return prompt

# Function to standardize one batch
def standardize_batch(batch):
    prompt = build_prompt_batch(batch)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for facility name standardization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        results = []
        for line in content.split("\n"):
            if "." in line:
                cleaned = line.split(".", 1)[1].strip()
                results.append(cleaned)

        # Ensure output length matches input
        if len(results) < len(batch):
            missing = len(batch) - len(results)
            results.extend(["[ERROR: missing]"] * missing)
        elif len(results) > len(batch):
            results = results[:len(batch)]

        return results
    except Exception as e:
        return [f"[ERROR: {e}]"] * len(batch)

# Process the data in batches
final_results = []
batch_size = 10

for i in range(0, len(facilities), batch_size):
    batch = facilities[i:i+batch_size]
    print(f"Processing rows {i+1} to {i+len(batch)}...")
    cleaned_batch = standardize_batch(batch)
    final_results.extend(cleaned_batch)
    time.sleep(1.5)

# Ensure the result length matches the dataframe
if len(final_results) != len(df):
    print(f"Warning: Results length ({len(final_results)}) ≠ DataFrame length ({len(df)}). Truncating or padding...")
    if len(final_results) > len(df):
        final_results = final_results[:len(df)]
    else:
        final_results.extend(["[ERROR: missing]"] * (len(df) - len(final_results)))

# Save final results
df["llm_structured_output_final"] = final_results
df.to_excel(output_path, index=False)
print("Output file saved.")
