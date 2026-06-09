# AI-Powered Customer Inquiry Assistant for a Textile Company

## Project Overview

This project is a prototype of an AI-powered customer inquiry assistant for a textile company. The goal is to help potential international customers ask questions about the company’s products, services, sustainability focus, production support, and inquiry process.

The assistant uses a simple Retrieval-Augmented Generation (RAG) workflow. Instead of answering only from general AI knowledge, the system first retrieves relevant company information from a structured knowledge base and then generates a grounded answer based on that retrieved context.

This project is designed for a Generative AI for Business Analytics course and demonstrates how company knowledge, chunking, retrieval, and grounded answer generation can be used in a practical business use case.

---

## Business Problem

International customers may have questions about textile products, denim fabric, customization, quality control, sustainability, pricing, shipping, and production support. However, customers may not always know where to find this information on a company website or how to ask the right questions.

This project explores whether a RAG-based assistant can help answer these customer inquiries clearly and consistently using company knowledge content.

---

## Data Used

This project uses two types of data.

### 1. Company Knowledge Base

File:

```text
company_info_sample.txt