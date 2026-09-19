PYTHON ?= python
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: audit test dry-run

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

dry-run:
	$(PYTHON) scripts/paper_clean_a16_campaign.py parent --datasets ETTh1 --seeds 2021 --dry-run
	$(PYTHON) scripts/local_spectrum_tangent.py select --datasets ETTh1 --seeds 2021 --widths 4 --dry-run

audit: test dry-run
	$(PYTHON) -m compileall -q data_provider exp layers models scripts tests utils run.py
	$(PYTHON) scripts/check_release.py
