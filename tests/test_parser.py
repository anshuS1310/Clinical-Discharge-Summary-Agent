# tests/test_parser.py
import unittest
from unittest.mock import MagicMock
from src.parser import ClinicalTextParser

class TestClinicalTextParser(unittest.TestCase):
    def setUp(self):
        self.parser = ClinicalTextParser()

    def test_filter_relevant_pages(self):
        # Mock a PDF reader with 3 pages
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(), MagicMock(), MagicMock()]
        
        # Page texts:
        # Page 1: Clinical (discharge summary)
        # Page 2: Administrative (room tariff, billing clearance)
        # Page 3: Sparse/Scanned (needs OCR, under 50 chars)
        page_texts = [
            "PATIENT DEMOGRAPHICS: Name: Jane Doe | Chief Complaint: Fever. Discharge summary.",
            "Inpatient bill statement. Visitor pass details. Room tariff consent clearance.",
            "Sparse text"
        ]
        
        relevant_indices = self.parser._filter_relevant_pages(mock_reader, page_texts)
        
        # Page 2 (index 1) should be filtered out
        self.assertIn(0, relevant_indices)
        self.assertNotIn(1, relevant_indices)
        self.assertIn(2, relevant_indices)

    def test_detect_patient_name(self):
        # Standard format
        text1 = "Patient Name: John Smith | Pt ID: SSS123 | Age: 45"
        self.assertEqual(self.parser._detect_patient_name(text1), "John Smith")
        
        # spelling variation normalization for demo patients
        text2 = "Pt. Name: PR EMA J. | Age: 30"
        self.assertEqual(self.parser._detect_patient_name(text2), "Prema J")
        
        # Blacklisted names should be rejected
        text3 = "Patient Name: STAFF NURSE | Bed: 302"
        self.assertEqual(self.parser._detect_patient_name(text3), "")

    def test_split_records_by_patient(self):
        page_texts = [
            "Patient Name: John Doe\nAdmission Note...",
            "Page 2 containing clinical history of John Doe...",
            "Pt. Name: Jane Smith\nDischarge Advice...",
            "Page 4 containing clinical course of Jane Smith..."
        ]
        
        records = self.parser._split_records_by_patient(page_texts, "test.pdf")
        
        # Verify that pages are grouped correctly
        self.assertIn("John Doe", records)
        self.assertIn("Jane Smith", records)
        self.assertIn("[Source page 1]", records["John Doe"])
        self.assertIn("[Source page 2]", records["John Doe"])
        self.assertIn("[Source page 3]", records["Jane Smith"])
        self.assertIn("[Source page 4]", records["Jane Smith"])

if __name__ == "__main__":
    unittest.main()
