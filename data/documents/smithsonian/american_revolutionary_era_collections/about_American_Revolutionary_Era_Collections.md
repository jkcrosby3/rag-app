# Smithsonian Revolutionary-era Collections

## Dataset Card for Smithsonian American Revolutionary Era Collections

### Links

- [Link](https://huggingface.co/datasets/RevolutionCrossroads/si_us_revolutionary_era_collections)

from datasets import load_dataset
 [Dataset](ds = load_dataset("RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs"))

## RevolutionCrossroads/si_images_textlabeling_bah

### Links

- [Link](https://huggingface.co/datasets/RevolutionCrossroads/si_images_textlabeling_bah)

import pandas as pd

[Dataset](
df = pd.read_parquet("hf://datasets/RevolutionCrossroads/si_images_textlabeling_bah/dataset.parquet"))

## Dataset Card for Smithsonian American Revolutionary Era Collections

### Dataset Summary

A specially selected subset of the Smithsonian’s Open Access collections covering objects from 1770–1810 selected for the Revolution Crossroads project in honor of the 250th anniversary of the founding of the United States. Drawn from four museums—the National Museum of American History, National Postal Museum, Smithsonian American Art Museum, and National Portrait Gallery—the dataset includes 12,667 records with descriptive metadata and nearly 4,000 linked images. It provides a foundation for exploring Revolutionary-era material culture and developing multimodal research and analysis tools.

### Dataset Description

The Smithsonian Revolutionary Era Collections dataset is a specially selected subset of the Smithsonian’s Open Access corpus, created as part of the Revolution Crossroads project in honor of the 250th anniversary of the United States.

It brings together descriptive records and related images from across four Smithsonian museums—the National Museum of American History, the National Postal Museum, the Smithsonian American Art Museum, and the National Portrait Gallery—that roughly date to the period 1770–1810, the era of the American Revolution and early republic.

The dataset includes: - 12,667 records describing museum objects and archival items - Metadata fields converted from Smithsonian Open Access JSON to Parquet format for analysis - 3,941 records linked to digital surrogates (thumbnail images and full object files)

This dataset was assembled to support research and experimentation at the intersection of cultural heritage and artificial intelligence. It provides a structured corpus for exploring American Revolutionary-era material culture, testing multimodal approaches that link descriptive metadata with images, and developing tools for analysis, visualization, and discovery.

Dataset Details
Prepared by: Smithsonian Institution, Office of Digital & Innovation staff
Shared by: Revolution Crossroads
Language(s): English
License: CC0-1.0
Dataset Sources
Repository: Smithsonian Open Access, Smithsonian Institution
Records contributed by:
National Museum of American History
National Postal Museum
Smithsonian American Art Museum
National Portrait Gallery
Access Path: All records were retrieved through the Smithsonian Open Access API
Curation Rationale
This dataset was curated to support the Revolution Crossroads project, which explores the Revolutionary and early republic period in recognition of the 250th anniversary of the United States.

The goal was to assemble a cross-museum dataset that: - Highlights objects dating between 1770–1810, the years spanning the American Revolution and early republic - Brings together holdings from across Smithsonian museums to allow comparative and cross-collection analysis - Focuses on descriptive metadata and images, fields most likely to support research, teaching, and machine learning applications - Provides a manageable, structured corpus for use in AI and digital humanities experimentation, rather than exposing the full complexity of Smithsonian Open Access

Because of these qualities, the dataset lends itself to a range of applications, including: - Exploring late Colonial and early American material culture through museum object metadata and images - Studying connections between objects, people, places, and themes represented in Smithsonian collections - Developing multimodal tools that combine descriptive metadata with linked images - Training and evaluating models for image classification, metadata-based clustering, or information retrieval in cultural heritage contexts - Supporting educational and public history projects that highlight Smithsonian holdings from the Revolutionary era

Dataset Creation
The Smithsonian Revolutionary Era Collections dataset was created by identifying objects and artwork records dated to the period 1770–1810 across four Smithsonian museums, in collaboration with curators and collections staff. Once the target lists of objects were compiled, the corresponding records were retrieved through Smithsonian Open Access.

National Postal Museum Smithsonian American Art Museum National Portrait Gallery - These museums were able to generate relatively complete listings of Revolutionary-era holdings.

National Museum of American History (NMAH) - NMAH records required additional review. - Date fields are free-text and uncontrolled, making searching difficult. - As a result: - Some non-period materials were swept up in initial searches (e.g., objects about the Revolutionary era but created later) - Manual cleanup was used to exclude broad ranges that fell mostly outside of scope (e.g., “1800–1900”) and replicas/commemoratives where identifiable - Some true period objects may still be missing due to incomplete or provisional records

Notes on Completeness
The dataset is not comprehensive. It represents the subset of Revolutionary-era records identifiable at the time of creation, filtered with curatorial input. Future cataloging work may surface additional relevant objects.

Data Collection and Processing
Processing Steps
Retrieved JSON records from Open Access for each identified object
Flattened and converted to Parquet format for efficient analysis
Retained only fields most relevant to anticipated uses (e.g., title, date, object type, place, names, topics, identifiers, media references)
Preserved mediaCount, mediaURLs, and thumbnail fields to allow linking to images
Maintained free-text contextual fields (e.g., date, objectType, name) alongside parallel indexed_* fields for normalized values
Supporting Files Available
JSON-formatted data from Open Access are available in the Files tab:

"api_json" - contains the JSON output of the records from the dataset, stored in directories for each of the 4 museums included. The file naming convention is "[unit]_[object_number].json", with the object number and unit abbreviation from the records. In the main dataset, the object_number may be found in the "EDANid" fields after "edanndm:".
.jpg format media files retrieved via Open Access are available in the Files tab:

"media" contains .jpg files for all media from the included records. The file names are unchanged from retrieval and reflect different naming practices of each museum, but generally include the museum acronym, an ID number, and often an appended number or letter if there are multiple images for a single record. In the main dataset, the associated image file names may be found as "id" in the "media_url" fields.
Quality Considerations
The dataset was not normalized or corrected
Some non-period records may remain, and some relevant ones may be missing
Metadata inconsistencies reflect differences in cataloging practice across museums
Dataset Structure
The dataset consists of two main components. Descriptive Records - 12,667 records representing museum objects and archival items - Originally in JSON via Smithsonian Open Access, converted to Parquet for analysis and ease of loading - Each record corresponds to a single object/item - Includes Smithsonian metadata fields such as title, description, date, place, subject terms, medium, dimensions, and credit line - Each record has a catalog number or identifier (not guaranteed unique)

Linked Images - 3,941 records include digital surrogates (thumbnail images and links to full object files) - Images are referenced by URL and can be retrieved directly from Smithsonian servers - Image file names and URLs are tied to their parent object record via identifier

File Organization - Parquet files contain the flattened descriptive metadata - Image links are stored within the metadata fields; no image files are bundled in the dataset - The dataset can be loaded directly into a dataframe-style environment for analysis, with image references resolvable via URL

Data Fields
Core Descriptive Fields
collectionsURL (string)
Public Smithsonian Collections record URL.
Example: <https://collections.si.edu/search/detail/edanmdm:nmah_1316741>

unitCode (string) Contributing museum code.
Example: NMAH

dataSource (string)
Human-readable name of the contributing museum/unit.
Example: National Museum of American History

title (string)
Cataloged title of the object.
Example: George Washington commemorative medal

EDANid (string)
EDAN record identifier (not a URL).
Example: edanmdm:nmah_1316741

guid (string)
Persistent GUID/ARK resolver URL.
Example: <http://n2t.net/ark:/65665/ng49ca746b1-4fbb-704b-e053-15f76fa0b4fa>

recordLink (string)
Unit website record URL. May not be present for all records - use guid instead. Example: <https://postalmuseum.si.edu/object/npm_2024.2006.0.1>

lastUpdateDate (date)
Date the record was last updated in Open Access.
Example: 2023-07-15

Descriptive Content
creditLine (string)
Credit or source line, often with a contextual label.
Example: Credit Line: Gift of Mrs. John Doe, 1905

date (string)
Date information with contextual labels.
Example: Date made: 1776

identifier (string)
Catalog/reference identifiers with labels (not guaranteed unique).
Example: Accession Number: 1978.1234

name (string)
Names of associated individuals with contextual labels.
Example: Maker: Paul Revere; Sitter: George Washington

notes (string)
Additional descriptive or catalog notes with labels.
Example: Exhibition: On view at the National Museum of American History

objectType (string)
Object category or classification with labels.
Example: Object Type: Broadside

physicalDescription (string)
Medium/materials/dimensions with labels.
Example: Medium: Oil on canvas; Dimensions: 30 ½ × 25 ¼ in.

place (string)
Place(s) associated with the object, with labels.
Example: Place made: Philadelphia, Pennsylvania

publisher (string)
Publisher information for printed works.
Example: Publisher: John Dunlap, Philadelphia

topic (string)
Subject terms or topical keywords.
Example: Topic: American Revolution

Indexed Fields (normalized values)
Normalized versions of descriptive fields (e.g., indexed_names, indexed_places). These lack contextual labels and provide standardized values.

indexed_dates (list[string])
Standardized dates extracted from free-text fields.
Example: [1776, 1790s]

indexed_names (list[string])
Controlled-format names.
Example: [Revere, Paul; Washington, George]

indexed_object_types (list[string])
Normalized object classifications.
Example: [broadside, medal]

indexed_places (list[string])
Standardized place names.
Example: [Philadelphia (Pa.), London (England)]

indexed_topics (list[string])
Normalized subject keywords.
Example: [American Revolution, Independence, Portraits]

Media Fields
mediaCount (integer)
Number of media items linked to the record.
Example: 2

mediaURLs (list[string])
URLs for digital surrogates.
Example:
["https://ids.si.edu/ids/deliveryService?id=NMAH-78-12345", "https://ids.si.edu/ids/deliveryService?id=NMAH-78-12346"]

thumbnail (string)
URL for a preview thumbnail image.
Example:
<https://ids.si.edu/ids/deliveryService?id=NMAH-78-12345&max=200>

Source Data
Original Data Producers
Records were created and are maintained by staff at: - National Museum of American History (NMAH) - National Postal Museum (NPM) - Smithsonian American Art Museum (SAAM) - National Portrait Gallery (NPG)

Data Access
Records were accessed via Smithsonian Open Access after the object lists were confirmed with museum staff
Open Access provided a standardized way to pull complete records across multiple units
Characteristics of Source Records
Metadata reflects cataloging practices over many decades and may have changed over time
Some fields are structured (e.g., accession numbers), while others are free-text with contextual labels (e.g., “date made: circa 1775”)
Cataloging depth varies by museum and by object: some records are richly detailed, others more minimal
Personal and Sensitive Information
There is no known personal or sensitive information included in this dataset.

Considerations for Using the Data
Risks and Limitations
This dataset represents a curated subset of Smithsonian collections rather than a complete record of American Revolutionary-era holdings. - Some Revolutionary-era objects may not be included if their catalog records lacked sufficient detail or were not discoverable through metadata searches - Some objects outside the 1770–1810 period may be present if their records used broad or ambiguous date ranges (e.g., “1800–1900”) - Metadata fields are descriptive but not always comprehensive. Some records may be incomplete if they have not yet been fully researched or updated

The original records in Smithsonian Open Access were created over many decades by different units and staff, leading to variation in terminology, structure, and level of detail. Cataloging is an ongoing process, and records are continually updated as new research is conducted.

For additional reuse limitations, users should consult the Smithsonian Open Access FAQ and Smithsonian Terms of Use.

Recommendations
Users should be aware of the risks and limitations of the dataset. Additional guidance may be added as further recommendations are developed.

Additional Information
Citation Information
BibTeX @dataset{RevolutionCrossroads_Smithsonian_2025, author = {Revolution Crossroads Project Team}, title = {Smithsonian American Revolutionary Era Collections}, year = {2025}, publisher = {Hugging Face}, url = {<https://huggingface.co/datasets/RevolutionCrossroads/si_us_revolutionary_era_collections}>, doi = {10.57967/hf/6527} }

APA Revolution Crossroads Project Team. (2025). Smithsonian American Revolutionary Era Collections [Data set]. Hugging Face. <https://doi.org/10.57967/hf/6527>

Glossary
EDAN (Enterprise Digital Asset Network): Smithsonian’s internal collections database system. Each object or item has an EDAN identifier
Indexed fields: Normalized versions of descriptive fields (e.g., indexed_names, indexed_places). These lack contextual labels (like “Maker” or “Sitter”) and provide just the standardized value, making them easier for search and filtering
Open Access: Smithsonian’s public data service that provides standardized access to digitized collections records and media
Thumbnail: A small preview image linked to an object, often used to quickly display collections online
Parquet: A columnar data file format optimized for efficient analysis and storage. Used here to make large metadata files easier to work with
Dataset Card Contact
<revolutioncrossroads@si.edu>

===================================================================

# RevolutionCrossroads/nara_revolutionary_war_pension_files

- [Link](https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files)

from datasets import load_dataset

- [Dataset](ds = load_dataset("RevolutionCrossroads/nara_revolutionary_war_pension_files"))

Dataset Card for American Revolutionary War Pension Files
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
