#!/usr/bin/env python3
"""
Standalone script to test coverage targets.
Run this directly to get the coverage analysis report.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Change to the workspace directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Now we can import the test module
from tests.test_coverage_targets import TestCoverageDetailedAnalysis

if __name__ == '__main__':
    tester = TestCoverageDetailedAnalysis()
    tester.test_all_graphs_detailed_report()
