# Revolutionary War Database Terminology

## Military Personnel Categories

### Soldiers (Enlisted Personnel)
- **Private**: Basic enlisted rank, most common in pension files
- **Corporal**: Non-commissioned officer, squad leader
- **Sergeant**: Senior non-commissioned officer, platoon leader
- **Drummer/Fifer**: Musicians who provided battlefield communication

### Officers (Commissioned Personnel)
- **Ensign**: Lowest commissioned officer rank
- **Lieutenant**: Junior officer (2nd Lieutenant, 1st Lieutenant)
- **Captain**: Company commander
- **Major**: Battalion second-in-command
- **Lieutenant Colonel**: Battalion commander
- **Colonel**: Regiment commander
- **Brigadier General**: Brigade commander
- **Major General**: Division commander
- **General**: Army commander (e.g., George Washington)

### Veterans
- **Veteran**: Any soldier or officer who served and survived the war
- **Pensioner**: Veteran who applied for and received a pension
- **Widow**: Surviving spouse of a veteran, eligible for pension benefits

## Document Collections

### Pension Files (12,606 documents)
- **Revolutionary War Pension and Bounty-Land Warrant Application Files**
- Each file represents ONE unique veteran (soldier or officer)
- Contains:
  - Service records (dates, units, battles)
  - Pension applications
  - Affidavits from fellow soldiers
  - Correspondence with Pension Office
  - Widow/heir claims
- **Total Veterans Represented**: ~12,606 soldiers and officers

### Newspapers (65 documents)
- Historical newspaper issues from Revolutionary War era
- Primary source news accounts
- Contemporary perspectives on events

### Revolutionary Era Collections (12,641 documents)
- Historical documents from the Revolutionary period
- Letters, diaries, official records
- Additional primary sources

### Smithsonian Collections (25,312 documents)
- General Smithsonian historical documents
- May include artifacts, correspondence, administrative records
- Broader historical context

## Database Structure

### Chunks vs Documents
- **Document**: One complete source file (e.g., one pension application)
- **Chunk**: A segment of a document for embedding (average 2.6 chunks per pension file)
- **Total Chunks**: 90,176 embedded text segments
- **Total Unique Documents**: ~50,624 unique source files

### Metadata Fields
- `topic`: Collection category (pension_files, newspapers, rev_era_collections, smithsonian)
- `file_name`: Original filename (e.g., 111403815.txt)
- `file_path`: Full path to source document
- `relative_path`: Path relative to data/documents/smithsonian/
- `chunk_index`: Which chunk of the document (0-indexed)
- `total_chunks`: How many chunks the document was split into

## Query Tips

### To find soldiers/officers:
- "pension application soldier name rank regiment"
- "veteran served continental army revolutionary war"
- "officer captain major colonel revolutionary war"

### To get counts:
- Use the **Statistics tab** for accurate totals
- RAG queries only retrieve 20-50 documents at a time
- Cannot count all 12,606 soldiers through RAG alone

### To search specific collections:
- Mention "pension files" for veteran records
- Mention "newspaper" for contemporary accounts
- Mention "revolutionary era" for broader historical documents

## Important Notes

1. **Each pension file = One veteran**: The 12,606 pension files represent approximately 12,606 unique Revolutionary War soldiers and officers

2. **Officers are included**: When asking about "soldiers," the database includes both enlisted soldiers AND commissioned officers unless you specifically exclude officers

3. **Terminology matters**: 
   - "Veterans" = soldiers + officers who survived
   - "Soldiers" typically = enlisted personnel only
   - "Military personnel" = all ranks (soldiers + officers)
   - "Pensioners" = those who applied for pensions

4. **RAG limitations**: 
   - Cannot count all documents through queries
   - Retrieves most relevant 20-50 chunks
   - Use Statistics tab for accurate totals
