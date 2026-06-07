# Revolutionary War Pension Files

## RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs

[Link](https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs)

from datasets import load_dataset

[dataset](ds = load_dataset("RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs"))

### Dataset Card for American Revolutionary War Pension Files - File-Level

Dataset Summary
A dataset derived from the National Archives and Records Administration (NARA) series Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service (NARA Catalog Series, NAID 300022). This dataset provides a file-level representation of Revolutionary War pension records, aggregating individual page records into complete pension files with associated metadata, extracted text, and transcriptions.

It is derived from the page-level dataset:
<https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files>

This dataset was prepared as part of the Revolution Crossroads project:
<https://www.si.edu/revolution-crossroads>

Dataset Description
For the initiative Revolution Crossroads, the Smithsonian Institution prepared this dataset using data, metadata, and digital objects publicly available from the National Archives Catalog.

Dataset Details
Prepared by: Smithsonian Institution, Office of Digital & Innovation staff
Shared by: Revolution Crossroads
Language(s): English, some French
License: Public domain
Relationship to Source Dataset
This dataset is a transformation of the original Revolution Crossroads page-level dataset. For detailed information about:

source materials
digitization processes
metadata definitions
rights and provenance
refer to the page-level dataset card:
<https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files/blob/main/README.md>

Curation Rationale
The original dataset is structured at the page level, where each record represents a single page from a pension file. However, pension files are multi-page archival documents that often contain narratives, affidavits, correspondence, and supporting materials that span multiple pages.

This file-level dataset was created to:

Preserve document-level context across entire pension files
Support OCR and text extraction workflows at the file level
Enable analysis of relationships between individuals, places, and events within a single case file
Support entity extraction and historical interpretation across full narrative records
Dataset Creation
This dataset was derived from the page-level pension dataset by restructuring records from page-level to file-level.

Page-level records were grouped by NAID (file-level identifier)
File-level records retain key metadata from the source dataset
Page counts were calculated for each file (numberOfPages)
Page-level data (images, extracted text, and transcriptions) were retained as lists within each file-level record
Some page-level metadata fields not relevant to file-level analysis were removed
A note on archival levels:

In most cases, records correspond to file-level groupings. However, a small number of records represent an intermediate “item” level between file and page in the NARA Catalog hierarchy
These item-level records were treated equivalently to file-level records in this dataset
Any record containing one or more pages was used as the top-level grouping unit (NAID), regardless of whether it was formally classified as a file or item
Each record in this dataset corresponds to a single pension file or equivalent document-level unit.

Data Collection and Processing
Supporting Files Available
File-level PDF files are provided in the Files tab for this dataset, within the media directory.

pdfs
Contains compiled PDF files representing complete pension files. Each PDF corresponds to a single record in the dataset and is constructed from the source page images.
Note:
pdfObjectID and pdfURL fields refer to PDFs provided by the National Archives Catalog where available. These may be incomplete or inconsistently available. The supplementary PDFs included with this dataset were generated to provide consistent, complete file-level representations.

PDF Construction
Source page images were obtained as JPG files from the NARA Catalog
Images were grouped by top-level NAID and compiled into multi-page PDFs
PDFs were constructed to represent complete pension files
To ensure compatibility with downstream processing systems:

Some PDFs were resized when they exceeded size limits during OCR extraction workflows
Resizing was applied selectively to affected files
A future update will provide consistently resized PDFs for broader external use
Dataset Structure
Each record in the dataset corresponds to a single pension file. File-level metadata is represented once per record, while page-level data (images, text, transcriptions) are stored as lists.

Data Fields
NAID (string)
National Archives Identifier for the pension file.
Example:
111403815

naraURL (string)
Link to the pension file in the National Archives Catalog.
Example:
<https://catalog.archives.gov/id/111403815>

title (string)
Title of the pension file.
Example:
Revolutionary War Pension and Bounty Land Warrant Application File B. L. Wt. 2,153-400, Joseph Torrey

logicalDate (string)
Normalized or machine-readable date, if available.
Example:
1836-06-02

pageObjectId (list)
List of identifiers for individual pages within the file.
Example:
["111403816", "111403817"]

pageURL (list)
List of URLs to page images.
Example:
["https://s3.amazonaws.com/NARAprodstorage/.../image1.jpg"]

numberOfPages (int64)
Number of pages in the pension file.
Example:
5

pageImageType (string)
File format of the page images.
Example:
JPG

extractedTextID (list)
Identifiers for extracted text records.
Example:
["19537435", "19537436"]

extractedText (list)
Extracted text for each page, provided via automated OCR/AI processes.
Example:
["State of New York...", "City and County of New York..."]

extractedTextDate (list)
Dates when extracted text was created.
Example:
["2024-11-20T13:30:07.000Z"]

extractedTextContributor (string)
Source of the extracted text.
Example:
FamilySearch

transcriptionID (list)
Identifiers for transcription records, if available.
Example:
["60f6e82a-f53d-4de3-a09b-f4f707046c0f"]

transcriptionText (list)
Human-created transcription text for pages, where available.
Example:
["Gorham Me March 22 1868..."]

transcriptionDate (list)
Dates when transcriptions were created or updated.
Example:
["2025-04-30 17:15:44"]

pdfURL (string)
Redirect URL to the file-level PDF as stored on Hugging Face in this dataset.
Example:
<https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs/resolve/main/media/pdfs/54961568-111412574/111403760.pdf>

Source Data
This dataset is derived from the National Archives and Records Administration series:
<https://catalog.archives.gov/id/300022>

For full details on the source collection, refer to the page-level dataset card:
<https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files/blob/main/README.md>

Personal and Sensitive Information
No known personal or sensitive information is included beyond what appears in historical archival records. These materials may contain outdated or offensive terminology reflective of the period in which they were created.

Considerations for Using the Data
Risks and Limitations
Pension files vary widely in length, structure, and completeness
Extracted text accuracy varies, especially for handwritten documents
Transcriptions are available for only a portion of records
Metadata may be incomplete or inconsistent
Recommendations
Researchers should validate extracted information against source images when accuracy is critical and consult the National Archives Catalog for the most up-to-date records.

Additional Information
Citation Information
BibTeX

@misc{revolution_crossroads_2026,
    author       = { Revolution Crossroads },
    title        = { nara_revolutionary_war_pension_files_PDFs (Revision 8453ff3) },
    year         = 2026,
    url          = { <https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs> },
    doi          = { 10.57967/hf/8349 },
    publisher    = { Hugging Face }
}

APA

Revolution Crossroads Project Team. (2025). American Revolutionary War Pension Files - File-Level [Data set]. Hugging Face.

Glossary
NAID: National Archives Identifier
Extracted text: Machine-generated text created from images
Transcription: Human-created text
Pension file: A complete archival case file documenting a pension application
Dataset Card Contact
<revolutioncrossroads@si.edu>

====================================================================

## RevolutionCrossroads/nara_revolutionary_war_pension_files

[Link](https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files)

from datasets import load_dataset

[dataset](ds = load_dataset("RevolutionCrossroads/nara_revolutionary_war_pension_files"))

### Dataset Card for American Revolutionary War Pension Files

Dataset Summary
A dataset derived from the Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service, ca. 1800–ca. 1912 (NARA Catalog Series, NAID 300022). This dataset includes page-level records with digitized images, extracted text, and human-created transcriptions where available. It offers a unique window into the lives of veterans and their families in the decades following the American Revolutionary War and provides a foundation for research, machine learning, genealogy, and public history projects.

Dataset Description
In honor of the 250th anniversary of the founding of the United States, the Smithsonian Institution prepared this dataset for the Revolution Crossroads initiative using data, metadata, and digital objects publicly available from the National Archives and Records Administration Catalog.

The dataset is derived from the National Archives and Records Administration series Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service, ca. 1800–ca. 1912 (NAID 300022). This series includes more than 82,000 case files containing over 2 million digitized pages.

The pension files preserve personal stories and supporting documents from veterans and their families who applied for pensions or land grants after the American Revolutionary War. These records often include details such as where and when a veteran served, and in which battles they took part. Many files also include applications from widows seeking benefits, offering glimpses into marriages, family life, and community ties.

The dataset includes:

2.2 million page-level records describing digitized pages from pension and bounty-land warrant case files
Metadata fields exported from the National Archives Catalog and converted to Parquet format for analysis
Links to digital surrogates (page images and multi-page PDFs) with persistent identifiers
Extracted text for all records and human-created transcriptions for approximately 27 percent of files
This dataset was prepared to support research and experimentation at the intersection of cultural heritage and artificial intelligence. It provides a structured corpus for examining the personal and social history of the Revolutionary era, testing methods for handwriting recognition and transcription, and developing tools for large-scale text analysis, visualization, and discovery. Together, these documents provide a rich and sometimes deeply personal view of life during and after the American Revolutionary War.

Dataset Details
Prepared by: Smithsonian Institution, Office of Digital & Innovation staff
Shared by: Revolution Crossroads
Language(s): English, some French
License: Public domain

Dataset Source
Repository: National Archives Catalog, National Archives and Records Administration
Series: Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service, ca. 1800–ca. 1912
National Archives Identifier: 300022
Record Group: Records of the Department of Veterans Affairs, 1773–2007

Curation Rationale
This dataset was prepared as part of the Revolution Crossroads project to enable large-scale analysis and reuse of Revolutionary War pension records in recognition of the 250th anniversary of the United States.

Revolution Crossroads aims to make available a structured dataset that:

Provides page-level access to one of the most important archival series documenting the aftermath of the American Revolutionary War
Brings together images, extracted text, and transcriptions for multi-modal analysis
Supports study of handwriting recognition, transcription validation, and information retrieval at scale
Creates opportunities for public engagement, genealogy, and family history research
Because of these qualities, the dataset lends itself to a range of applications, including:

Exploring the social and personal history of veterans and their families in the early republic
Studying connections between individuals, places, and events documented in the files
Evaluating and improving OCR and transcription methods for handwritten documents
Developing large-scale text retrieval and entity recognition tools for digitized archival materials
Supporting genealogy, public history, and classroom projects related to the American Revolution
Dataset Creation
This dataset was assembled for hosting on Hugging Face by the Smithsonian Institution’s Office of Digital & Innovation staff as part of the Revolution Crossroads project. All records from the National Archives Catalog series Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service, ca. 1800–ca. 1912 (NAID 300022) were included.

The dataset was prepared by pulling JSON-formatted data from three NARA API endpoints – one for the record metadata, one for the extracted text, and one for the transcriptions (where available). These sources were combined and flattened so that each page is represented as a single record, and then stored in tabular Parquet format for efficient analysis and hosting on Hugging Face.

Data Collection and Processing
Processing Steps
Retrieved JSON-formatted data from three NARA API endpoints (records, extracted text, transcriptions)
Combined datasets into a single table keyed by record identifiers
Flattened into one record per line, with page-level granularity
Stored in Parquet format for efficient analysis and distribution via Hugging Face
Supporting Files Available
JSON-formatted data from three NARA API endpoints above are available in the Files tab:

"records/" contains the JSON output of the records from the datasets, stored in chunks of 1000 with the file naming convention "nara_results_[start]_[end].json", following the retrieval order of the NARA records API
"extracted_text/" contains CSV output of the extracted text of the records, stored in chunks of 1000 with the file naming convention "nara_results_[start]_[end]_extracted_text.csv", following the same retrieval order of the records API
"transcriptions/" contains JSON output of the transcriptions of the records, stored in chunks of 1000 with the file naming convention "nara_results_[start]_[end]_transcriptions.json", following the same retrieval order of the records API
Quality Considerations
The Hugging Face dataset reflects the full content of the Catalog export; no records were added or removed
Extracted text is present for all records, produced through automated processes including AI assistance (via FamilySearch partnership)
Human transcriptions were created through the NARA Citizen Archivist program; these currently cover approximately 27 percent of files
Metadata reflects the Catalog at the time of export and is updated by NARA on an ongoing basis
Dataset Structure
Record Level
Each record in the dataset corresponds to a single digitized page from a pension file.

Data Fields
Core Identifiers
NAID (string)
National Archives Identifier for the pension file.
Example: 111769430

naraURL (string)
Link to the pension file record in the National Archives Catalog.
Example: <https://catalog.archives.gov/id/111769430>

title (string)
Title of the pension or bounty land warrant application file, often including the veteran’s name and case type.
Example: Revolutionary War Pension and Bounty Land Warrant Application File S 33948, Wm Woodbury, Mass. Note on File Title Abbreviations

S — Survivor / Soldier
R — Rejected
W — Widow
B, BLW, B L Wt. — Bounty Land Warrant
OW — Old War
NA Acc — National Archives Accession number (placeholder when no pension file was located)
Notes on errors: Some letters (P, K, T, H, M) are mistranscriptions of the above. These are being corrected in the NARA Catalog but may still appear in this dataset until a future data pull.

Dates and Identifiers
logicalDate (string)
Normalized or machine-readable date, if available.
Example: 1831-06-20

variantControlNumbers (string/JSON)
Alternate identifiers linked to the file.
Example: [{"number": "Fold3 2018", "type": "Search Identifier"}]

File-Level Data
Note: pdfObjectID and pdfURL refer to a multi-page file that includes all pages for a given pension application. These values repeat across all pages in the same file.

pdfObjectID (string)
Identifier for the file-level PDF.
Example: 19850379

pdfURL (string)
URL to the full multi-page PDF.
Example: <https://s3.amazonaws.com/NARAprodstorage/lz/rediscovery/14974.pdf>

Page-Level Data
pageObjectId (string)
Unique identifier for a single page image.
Example: 19850379

pageURL (string)
URL to the digitized page image.
Example: <https://s3.amazonaws.com/NARA/RevWar/177212_00591.jpg>

pageImageType (string)
File format of the page image.
Example: JPG

Extracted Text (OCR)
extractedTextID (string)
Identifier for the extracted text object.
Example:
33948 Revy INVALID. File No. 33948 Index William Woodbury Sir Rev War Act: 18 March 18 ...

extractedText (string)
Extracted text created via OCR/AI processes. Present for all records.
Example excerpt:
“State of Pennsylvania, County of Chester, On this 5th day of March 1833 personally appeared before the court John Smith, aged seventy-eight years…”

extractedTextDate (string)
Date OCR text was uploaded.
Example: 2024-11-20T20:32:03.000Z

extractedTextContributor (string)
Source of the OCR text.
Example: FamilySearch

Transcriptions (Citizen Archivists)
transcriptionID (string)
Identifier for the transcription record, if one exists.
Example: null or a unique string for transcribed pages.

transcriptionText (string)
Human-created transcription text. Available for approximately 27 percent of files.
Example excerpt:
“That he enlisted in the month of April 1777 for the term of three years…”

transcriptionContributionCount (integer)
Number of transcription contributions for the page.
Example: 3

transcriptionUserNames (list[string])
Usernames of volunteers who contributed.
Example: ["archivist_jane", "historybuff77"]

transcriptionDate (string)
Date the transcription was created or last updated.
Example: 2024-11-20T20:32:03.000Z

Source Data
Series of Records
Case Files of Pension and Bounty-Land Warrant Applications Based on American Revolutionary War Service, ca. 1800–ca. 1912 (NAID 300022)

Record Group
Records of the Department of Veterans Affairs, 1773–2007

Microfilm Publications
M804 — Revolutionary War Pension and Bounty-Land Warrant Application Files

Coverage
This series documents the time period ca. 1775–ca. 1900

Arrangement
Arranged alphabetically by last name of veteran (soldier)

Record Creators
Department of the Interior. Bureau of Pensions (1849–1930) (most recent)
War Department. Office of the Secretary (1789–1947) (predecessor)
War Department. Military Bounty Lands and Pension Branch (ca. 1810–1815) (predecessor)

Custody History
This series was maintained by the Office of the Secretary of War until ca. 1810, by the Military Bounty Lands and Pensions Branch from ca. 1810–1815, and thereafter by the Bureau of Pensions. The records are currently held by the National Archives and Records Administration.

Known Gaps
There is a gap of 84 rolls of images missing from this series. These are currently being digitized by the National Archives and will be added to the Catalog upon completion.

Digitization Partnerships
The National Archives microfilm was originally digitized by Fold3 through a partnership agreement.

In April 2024, the Archives partnered with FamilySearch to run all images through an AI large language model to produce first-draft extracted text. These drafts were added to the images in the National Archives Catalog as partner-contributed text.

Volunteer Crowdsourcing
National Archives volunteers, called Citizen Archivists, have been copying extracted text into the Catalog transcription module, reviewing and cleaning up errors, and publishing the corrected versions. These human-created transcriptions are available for approximately 27 percent of files to date (September 2025).

Personal and Sensitive Information
No known personal or sensitive information is included. Users should be aware that records may contain outdated terminology reflecting the period in which they were created.

Considerations for Using the Data
Risks and Limitations
This dataset represents a large-scale export of pension file pages from the National Archives Catalog. It is not a comprehensive or definitive edition of the underlying archival series.

Some pension files are incomplete due to missing rolls of microfilm
Extracted text accuracy varies; errors are especially common for handwritten pages
Human transcriptions improve accuracy but are only available for about 27 percent of files
Names, places, and dates may appear with inconsistent spelling or formatting
Cataloging metadata is updated by NARA on an ongoing basis
Recommendations
Users should be aware of the limitations described above when using the dataset. It is recommended that researchers consult the National Archives Catalog for the most up-to-date versions of the records and validate extracted information against the underlying images whenever possible.

Additional Information
Citation Information
BibTeX

@dataset{RevolutionCrossroads_NARA_2025, author = {Revolution Crossroads Project Team}, title = {American Revolutionary War Pension Files}, year = {2025}, publisher = {Hugging Face}, url = {<https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files}>, doi = {10.57967/hf/6529} }

APA Revolution Crossroads Project Team. (2025). American Revolutionary War Pension Files [Data set]. Hugging Face. <https://doi.org/10.57967/hf/6529>

Glossary
NAID: National Archives Identifier, a unique reference number for archival series and files
Extracted text: Machine-generated text created from scanned images using optical character recognition or AI-assisted methods. Present for all records in this dataset
Transcription: Human-created text entered manually by volunteers or staff. Available for approximately 27 percent of files
Fold3: Commercial partner that digitized NARA microfilm
FamilySearch: Partner of the National Archives that created extracted text for the series
Citizen Archivist: NARA’s volunteer program for transcription and metadata improvement
Dataset Card Contact
<revolutioncrossroads@si.edu>
