import sys
sys.stdout.reconfigure(encoding="utf-8")

readme_parts = []

readme_parts.append("""# GT Forgery Detection — Complete Guide

> **Who is this for?**  
> You have basic Python knowledge but are new to machine learning.
> This document explains *everything* — what the data is, how the models work,
> how training happens, and what the numbers mean — in plain language with diagrams.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Dataset](#2-the-dataset)
3. [Forgery Types — What Are We Detecting?](#3-forgery-types)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Architecture](#5-model-architecture)
6. [Training](#6-training)
7. [Performance Metrics](#7-performance-metrics)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [File Map](#9-file-map)

---
""")
