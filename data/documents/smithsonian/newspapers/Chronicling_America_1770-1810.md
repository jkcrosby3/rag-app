# Chronicling America, 1770-1810

# RevolutionCrossroads/loc_chronicling_america_1770-1810_issues

* [Link](https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810_issues)

from datasets import load_dataset

* [Dataset](ds = load_dataset("RevolutionCrossroads/loc_chronicling_america_1770-1810_issues")

Dataset Card for Chronicling America Historic American Newspapers 1770–1810 - Issue-Level
Dataset Summary
A dataset drawn from the Library of Congress Chronicling America digital collection, part of the National Digital Newspaper Program (NDNP). This dataset provides an issue-level representation of the Chronicling America newspapers dataset, aggregating individual page records into complete newspaper issues with OCR text and publication metadata for newspapers published between 1770 and 1810.

It is derived from the page-level dataset:
<https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810>

This dataset was prepared as part of the Revolution Crossroads project:
<https://www.si.edu/revolution-crossroads>

Dataset Description
For the initiative Revolution Crossroads, the Smithsonian Institution prepared this dataset using data, metadata, and digital objects publicly available from the Chronicling America Historic American Newspapers collection.

Dataset Details
Prepared by: Smithsonian Institution, Office of Digital & Innovation staff
Shared by: Revolution Crossroads
Language(s): English
License: Public domain
Relationship to Source Dataset
This dataset is a transformation of the original Revolution Crossroads page-level dataset. For detailed information about:

source materials
digitization processes
metadata definitions
rights and provenance
refer to the page-level dataset card:
<https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810/blob/main/README.md>

Curation Rationale
The original dataset is structured at the page level, where each record represents a single newspaper page. However, newspapers are multi-page documents, and meaningful context often spans multiple pages within a single issue.

This issue-level dataset was created to:

Preserve document-level context across pages
Support OCR and text extraction workflows at the issue level
Enable entity recognition and analysis across full newspaper issues
Dataset Creation
This dataset was derived from the page-level Chronicling America dataset by restructuring records from page-level to issue-level.

Page-level records were grouped by issue (LCCN, issue date, edition_order)
Issue-level records retain key bibliographic metadata from the source dataset
Page counts were calculated for each issue (page_count)
An issue identifier (issue_id) was constructed using the format lccn_issue_date_edition_order
References to the source dataset were preserved
Each record in this dataset corresponds to a single newspaper issue.

Data Collection and Processing
Supporting Files Available
Issue-level PDF files are provided in the Files tab for this dataset.

pdfs
Contains compiled PDF files representing complete newspaper issues. Each PDF corresponds to a single record in the dataset and is constructed from the source page images. They are grouped into directories based on the newspaper (title) LCCN. Each pdf uses the naming format of the corresponding issue_id, with "titleLCCN_IssueDate_EditionNumber".
PDF Construction
Source page images were downloaded in JPEG2000 (JP2) format to preserve the highest available image quality
JP2 images were converted to JPG format for processing
The JP2 and JPG files did not contain DPI values. DPI values were derived from the source page-level PDFs (either 300 or 400 DPI) and applied when compiling images into issue-level PDFs
JPG images were assembled into a single PDF per issue
To ensure compatibility with downstream processing systems:

Images exceeding 10,000 pixels in height or 4,800 pixels in width were resized
Resizing was performed proportionally to preserve visual fidelity while maintaining a workable scale
Dataset Structure
Each record in the dataset corresponds to a single newspaper issue. Issue-level metadata (title, place, date, edition) is represented once per issue rather than repeated across pages.

Data Fields
lccn (string)
Library of Congress Control Number for the newspaper title.
Example:
sn82014385

issue_date (stringdate)
Date the issue was published.
Example:
1809-07-08

edition_order (string)
Edition number/order within the day. The default is 1.
Example:
1

issue_id (string)
Unique identifier for the issue derived from LCCN, date, and edition.
Example:
sn82014385_1809-07-08_1

newspaper_title (string)
Title of the newspaper.
Example:
The Delaware gazette

place_of_publication (string)
City and state of publication.
Example:
Wilmington [Del.]

page_count (int64)
Number of pages in the issue.
Example:
4

issue_url (string)
URL to the newspaper issue in Chronicling America.
Example:
<https://www.loc.gov/item/sn82014385/1809-07-08/ed-1>

iiif_manifest (string)
IIIF manifest URL for the issue.
Example:
<https://www.loc.gov/item/sn82014385/1809-07-08/ed-1/manifest.json>

thumbnail_url (list)
List of thumbnail image URLs for pages in the issue.
Example:
["https://tile.loc.gov/image-services/iiif/service:ndnp:.../0074/full/pct:6.25/0/default.jpg"]

jpeg2000_url (list)
List of JPEG2000 (JP2) image URLs for each page in the issue.
Example:
["https://tile.loc.gov/storage-services/service/ndnp/.../0074.jp2"]

pdf_url (string)
Redirect url to issue-level PDF as stored on Hugging Face in this dataset.
Example:
<https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810_issues/resolve/main/pdfs/sn82014385/sn82014385_1809-07-08_1.pdf>

ocr_url (list)
List of OCR XML file URLs for each page, as provided by Chronicling America contributing institutions.
Example:
["https://tile.loc.gov/storage-services/service/ndnp/.../0074.xml"]

ocr_text (list)
OCR text aggregated from all pages in the issue, derived from original Chronicling America OCR.
Example:
["THE DELAWARE GAZETTE. VOL. I. WILMINGTON, SATURDAY, JULY 8, 1809..."]

source_record_create_date (date32)
Date of source record creation.
Example:
2017-05-24

Source Data
This dataset is derived from the Chronicling America Historic American Newspapers collection collection from the Library of Congress.

For full details on the source collection, program context, and historical background, refer to the page-level dataset card:
<https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810/blob/main/README.md>

Personal and Sensitive Information
No known personal or sensitive information is included beyond what appeared in publicly circulated newspapers of the time. Users should be aware that newspapers may contain outdated or offensive terminology reflecting the period in which they were created.

Considerations for Using the Data
Risks and Limitations
The dataset contains historical materials with language that does not always match the language preferred by members of the communities depicted. It may include negative stereotypes or words that offend. These materials reflect the views of their creators, not the Library of Congress or the United States government.

Machine-readable text in the dataset was created using Optical Character Recognition (OCR). Errors are present throughout the data, particularly when source images are degraded or typography is unusual.

For content published before 1810, historical typesetting (e.g., “long s” characters resembling “f”) may result in misread text.

Metadata and digitized content are updated by the Library of Congress on an ongoing basis.

Recommendations
Researchers should validate extracted information against images when accuracy is critical and be cautious about treating OCR text as complete or authoritative. Consultation of the Chronicling America collection is recommended for the most up-to-date records.

Additional Information
Citation Information
BibTeX @misc{revolution_crossroads_2026, author = { Revolution Crossroads }, title = { loc_chronicling_america_1770-1810_issues (Revision 4b7dc29) }, year = 2026, url = { <https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810_issues> }, doi = { 10.57967/hf/8348 }, publisher = { Hugging Face } }

APA

Revolution Crossroads Project Team. (2025). Chronicling America: Historic American Newspapers 1770–1810 - Issue-level [Data set]. Hugging Face.

Glossary
LCCN: Library of Congress Control Number, a unique identifier for newspaper titles
OCR (Optical Character Recognition): Machine-generated text created from scanned page images
Issue: A single publication of a newspaper on a given date, often containing multiple pages
Dataset Card Contact
<revolutioncrossroads@si.edu>

=======================================================================

# RevolutionCrossroads/loc_chronam_textract_ocr_bah

Textract OCR data for the Chronicling America: Historic American Newspapers 1770–1810 dataset

[Link](https://huggingface.co/datasets/RevolutionCrossroads/loc_chronam_textract_ocr_bah)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset

[dataset](df = pd.read_parquet("hf://datasets/RevolutionCrossroads/loc_chronam_textract_ocr_bah/chronam_ocr_1770-1810.with_ocr_text_new.parquet")

# Login using e.g. `huggingface-cli login` to access this dataset

df = pd.read_parquet("hf://datasets/RevolutionCrossroads/loc_chronam_textract_ocr_bah/chronam_ocr_1770-1810.with_ocr_text_new.parquet"))

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset

[Dataset](df = pd.read_parquet("hf://datasets/RevolutionCrossroads/loc_chronam_textract_ocr_bah/chronam_ocr_1770-1810.with_ocr_text_new.parquet"))
Textract OCR data for the Chronicling America: Historic American Newspapers 1770–1810 dataset
Column descriptions and definitions (and more important documentation) can be found in the source dataset at <https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810>.

Dataset Structure
Record Level
Each record in the dataset corresponds to a single newspaper page. Issue-level metadata (title, place, date, edition) is repeated across all pages of that issue.

Data Fields
New columns in this derivative dataset are:

textract_OCR (string)
OCR results from the AWS Textract API

textract_date (datetime64)
The date and time that the Textract OCR was returned.

textract_source (string)
The source for the OCR data.

===============================================================

# RevolutionCrossroads/loc_chronicling_america_1770-1810

[Link](https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810)

[Dataset](df = pd.read_parquet("hf://datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810/chronam_ocr_1770-1810.parquet"))

Dataset Card for Chronicling America: Historic American Newspapers 1770–1810
Dataset Summary
A dataset drawn from the Library of Congress Chronicling America digital collection, part of the National Digital Newspaper Program (NDNP). This dataset includes page-level records with images, OCR text, and publication metadata for newspapers published between 1770 and 1810. It provides a foundation for research, machine learning, and public history projects in honor of the 250th anniversary of the United States.

Dataset Description
For the initiative Revolution Crossroads, the Smithsonian Institution prepared this dataset using data, metadata, and digital objects publicly available from the Chronicling America digital collection.

The dataset consists of text and images from newspapers published in the United States between 1770 and 1810. It was created by the Library of Congress as part of the National Digital Newspaper Program (NDNP), a partnership between the Library and the National Endowment for the Humanities (NEH).

The dataset includes:

Approximately 340,000 page-level records representing U.S. newspapers published between 1770 and 1810
Metadata fields exported from the Chronicling America API and converted to Parquet format for analysis
Links to digital surrogates (page images, thumbnails, PDFs, and OCR XML files) with persistent identifiers
Extracted text (OCR) for all records
This dataset was prepared to support research and experimentation at the intersection of cultural heritage and artificial intelligence. It provides a structured corpus for examining Revolutionary-era newspapers, testing OCR performance on historical typography, and developing tools for large-scale text analysis, visualization, and discovery.

Dataset Details
Prepared by: Smithsonian Institution, Office of Digital & Innovation staff
Shared by: Revolution Crossroads
Language(s): English
License: Public domain
Dataset Sources
Repository: Chronicling America, Library of Congress
Program: National Digital Newspaper Program (NDNP), a partnership between the Library of Congress and the National Endowment for the Humanities
Curation Rationale
This dataset was prepared as part of the Revolution Crossroads project in recognition of the 250th anniversary of the United States. It makes a large-scale collection of Revolutionary-era newspapers available in a structured format to enable a wide range of applications, including:

Studying the circulation of news and ideas in the early United States
Exploring themes, language, and public opinion during the Revolutionary era and early republic
Evaluating OCR quality on historical typefaces and 18th-century print conventions (e.g., “long s” characters)
Developing retrieval and discovery tools for large-scale historical text corpora
Supporting genealogy, public history, and educational projects that use newspapers as sources
Dataset Creation
This dataset was assembled for Hugging Face by the Smithsonian Institution’s Office of Digital & Innovation staff as part of the Revolution Crossroads project. Data was accessed from the Chronicling America “data” backend (<https://chroniclingamerica.loc.gov/data/>) to avoid API rate limits and reduce load on Library of Congress servers.

Batch information containing issue lists was retrieved, filtered to the target date range (1770–1810), and issue metadata was downloaded in MODSXML format along with OCR text in ALTOXML format. These sources were then combined and flattened so that each record corresponds to a single page of a newspaper issue, and stored in Parquet format for efficient analysis and hosting on Hugging Face.

Data Collection and Processing
Processing Steps
Accessed Chronicling America “data” backend (<https://chroniclingamerica.loc.gov/data/>) to retrieve batch information and issue lists
Filtered issues to the Revolutionary-era date range (1770–1810)
Downloaded issue metadata in MODSXML (bibliographic metadata) format and OCR text in ALTOXML (OCR text + structural information) format
Additional newspaper title metadata (newspaper_title and place_of_publication) was pulled from the Library of Congress API, based on the lccn list from the filtered issues.
Combined metadata and text into a single dataset keyed to issue/page identifiers
Flattened into one record per newspaper page
Stored in Parquet format for efficient analysis and hosting on Hugging Face
Supporting Files Available
XML-formatted data from data backend are available in the Files tab:

"mods_xml_records" - contains the MODS XML source files for the records from the dataset, stored in directories for each of newspaper lccns (title series numbers). The file naming convention is "[lccn][date][edition_order].xml".
"ocr_alto_xml" - contains the Alto XML source files for the records from the dataset, stored in directories for each of newspaper lccns (title series numbers). The file naming convention is "[date]_[edition_order].xml".
PDF versions of each page image are available in the Files tab:

"media" contains the pdf source files for the records from the dataset, stored in directories for each of newspaper lccns (title series numbers). The file naming convention is "[issue_date][edition_order][page].xml". -- please note, the jpeg2000 files, available via the links in the dataset, are of a higher resolution than the pdfs.
Quality Considerations
OCR accuracy varies depending on print quality, typography, and page condition
Historical typesetting, e.g., “long s” characters resembling “f”, or "ligatured ct" resembling a "d" or an "f", may result in misread text
Metadata reflects the Catalog at the time of export and is updated by the Library of Congress on an ongoing basis
Dataset Structure
Each record in the dataset corresponds to a single newspaper page. Issue-level metadata (title, place, date, edition) is repeated across all pages of that issue.

Data Fields
Core Identifiers
web_url (string)
URL to the newspaper issue or page in Chronicling America.
Example: [https://www.loc.gov/item/sn82016139/1776-07-15/ed-1/](https://www.loc.gov/item/sn82016139/1776-07-15/ed-1/)

lccn (string)
Library of Congress Control Number for the newspaper title.
Example: sn83045110

newspaper_title (string)
Title of the newspaper.
Example: The Pennsylvania Packet, and Daily Advertiser

Publication Metadata
place_of_publication (string)
City and state of publication.
Example: Philadelphia, Pennsylvania

issue_date (string)
Date the issue was published.
Example: 1777-05-14

edition_order (string)
Edition number/order within the day. The default is 1. Higher numbers indicate additional versions of the same issue published or digitized on that date. These may result from different scans of the same microfilm, rescans from original print, or intentional duplicates to improve OCR quality.
Example: 1

Page-Level Metadata
page (string)
Page number within the issue.
Example: 1
Media Fields
thumbnail_url (string)
URL to a thumbnail image of the page.
Example:
<https://tile.loc.gov/image-services/iiif/service:sgp:sgpbatches:misctop:batch_dlc_misctopsn82016139_ver01:data:sn82016139:print:1776071501:0004/full/pct:6.25/0/default.jpg#h=318&w=202>

pdf_url (string)
URL to the PDF of the page.
Example:
<https://tile.loc.gov/storage-services/service/ndnp/deu/batch_deu_kedavra_ver01/data/sn82014385/00271740232/1809071201/0079.pdf>

ocr_url (string)
URL to the ALTOXML OCR file for the page.
Example:
<https://tile.loc.gov/storage-services/service/ndnp/deu/batch_deu_kedavra_ver01/data/sn82014385/00271740232/1809071201/0079.xml>

Extracted Text (OCR)
ocr_text (string)
OCR text extracted from the page image. Present for all records.
Example excerpt:
"In CONGRESS, July 4, 1776. A DECLARATION by the Representatives of the UNITED STATES OF AMERICA..."
Record Creation
source_record_create_date (date)
Date of catalog record creation. Example excerpt:
2017-05-24
Source Data
Collection
The dataset is drawn from the Chronicling America Historic American Newspapers digital collection from the Library of Congress, which contains images, full text (OCR), and metadata for over 23 million digitized public domain newspaper pages originally published in the United States from the 1700s through 1963. Chronicling America is a product of the National Digital Newspaper Program (NDNP), a partnership between the National Endowment for the Humanities (NEH) and the Library of Congress.

Program
The Chronicling America Historic American Newspapers collection provides access to select digitized newspaper pages produced by the National Digital Newspaper Program (NDNP), a partnership between the National Endowment for the Humanities (NEH) and the Library of Congress. As part of the program, cultural heritage institutions apply for and receive awards from an NEH award program to select and digitize newspaper pages representing the history, geographic coverage, and events of note for their state or territory. Visit the Library of Congress' NDNP website for more information on program guidelines. As part of its role in the NDNP, the Library of Congress also contributes digitized newspaper pages from its own collections to Chronicling America.

Historical Background
Newspapers are frequently cited as being the “first draft" of U.S. history, capturing the day-to-day life of a community better than any other published record. With the goal of preserving this important format, from 1982 through 2011, the Library of Congress and the National Endowment for the Humanities collaborated to fund and manage the United States Newspaper Program (USNP), a highly successful effort to locate, catalog, and preserve newspapers published throughout the United States. USNP projects were established and funded in each state and territory to survey every possible repository in an attempt to locate extant issues of every newspaper, inventory and catalog those titles in a national database, and preserve endangered files on microfilm following national and international preservation standards.

In 2003, the NEH and the Library embarked on another partnership, the National Digital Newspaper Program (NDNP). The NDNP is a long-term effort that builds on the legacy of the USNP by increasing access through digitization to the valuable information generated by the USNP. The NDNP provides an opportunity for institutions to select and contribute digitized newspaper content to a freely accessible, national newspaper resource, Chronicling America.

Original Format
The newspapers in Chronicling America were originally published in newsprint, however participants in NDNP primarily scan from 35mm negative microfilm on which the newsprint was captured. Microfilm standards vary, but many historical newspapers in this dataset were microfilmed under the auspices of the United States Newspaper Program (USNP), 1982–2011. While the technical approach for NDNP digitization is primarily based on images scanned from negative microfilm, some amounts of NDNP batches may consist of images scanned from the original newsprint, positive microfilm, microfiche, or preservation facsimile copies, provided that technical specifications are followed. The original format type is specified in the metadata.

Source Data Producers
The source data (historical newspapers) was produced by people affiliated with the newspapers (reporters, editors, contributors, etc).

Rights
The Library of Congress believes that the newspapers in Chronicling America are in the public domain or have no known copyright restrictions. Newspapers published in the United States more than 95 years ago are in the public domain in their entirety. Responsibility for making an independent legal assessment of an item and securing any necessary permissions ultimately rests with persons desiring to use the item.

For more information on Library of Congress policies and disclaimers regarding rights and reproductions, see <https://www.loc.gov/legal/>

Personal and Sensitive Information
No known personal or sensitive information is included beyond what appeared in publicly circulated newspapers of the time. Users should be aware that newspapers may contain outdated or offensive terminology reflecting the period in which they were created.

Considerations for Using the Data
Risks and Limitations
The dataset contains historical materials with language that does not always match the language preferred by members of the communities depicted. It may include negative stereotypes or words that offend. These materials reflect the views of their creators, not the Library of Congress or the United States government. Historical materials may also contain factual errors and should be understood in the context of their particular time and place.

Machine-readable text in the dataset was created using Optical Character Recognition (OCR). OCR is an automated process that converts the visual image of text into machine-readable characters, allowing computer software to search for words, phrases, numbers, or other elements. While OCR is a powerful tool, errors in the process are unavoidable and are present throughout the data, particularly when source images are degraded or typography is unusual.

For content published before 1810, historical typesetting, e.g., “long s” characters resembling “f”, or "ligatured ct" resembling a "d" or an "f", may result in misread text."

Metadata and digitized content are updated by the Library of Congress on an ongoing basis.

Recommendations
Researchers should validate extracted information against images when accuracy is critical, and be cautious about treating OCR text as complete or authoritative. Consultation of the Chronicling America collection is recommended for the most up-to-date records.

Additional Information
Citation Information
BibTeX

@dataset{RevolutionCrossroads_LOC_2025, author = {Revolution Crossroads Project Team}, title = {Chronicling America: Historic American Newspapers 1770–1810}, year = {2025}, publisher = {Hugging Face}, url = {<https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810}>, doi = {10.57967/hf/6526} }

APA

Revolution Crossroads Project Team. (2025). Chronicling America: Historic American Newspapers 1770–1810 [Data set]. Hugging Face. <https://doi.org/10.57967/hf/6526>

Glossary
LCCN: Library of Congress Control Number, a unique identifier for newspaper titles
OCR (Optical Character Recognition): Machine-generated text created from scanned page images. Present for all records in this dataset
NDNP (National Digital Newspaper Program): A partnership between LOC and NEH to digitize historic newspapers
USNP (U.S. Newspaper Program): A precursor to NDNP, conducted 1982–2011 to inventory microfilmed newspapers nationwide
MODSXML: Metadata Object Description Schema, a Library of Congress XML format for bibliographic metadata (used for issue-level information in this dataset)
ALTOXML: An XML schema for storing OCR text and layout information (used for page-level OCR in this dataset)
Issue: A single publication of a newspaper on a given date, often containing multiple pages. In this dataset, issue-level metadata (title, place, date, edition) is repeated for each page of that issue.
Dataset Card Contact
<revolutioncrossroads@si.edu>
