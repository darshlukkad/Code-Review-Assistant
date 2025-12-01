# Test Suite

This directory contains all tests for the project.

## Test Files

- `test_data_loader.py` - Tests for data loading and preprocessing
- `test_model.py` - Tests for model architectures
- `test_training.py` - Tests for training pipeline
- `test_api.py` - Tests for FastAPI endpoints
- `test_inference.py` - Tests for model inference

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v tests/
```

## Test Coverage Goals

- Data loading: >90%
- Model forward pass: 100%
- API endpoints: >95%
- Inference: >90%

## TODO

- [ ] Implement unit tests for data loader
- [ ] Implement model tests
- [ ] Implement API integration tests
- [ ] Add test fixtures
- [ ] Set up CI/CD testing
