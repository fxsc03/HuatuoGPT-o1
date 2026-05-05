"""
Compatibility entrypoint for the accelerated verified Long_CoT pipeline.

The old monolithic implementation has been replaced by:
- `cot_pipeline_accelerated.py`: main verified Long_CoT synthesis pipeline
- `postprocess_verified_long_cot.py`: optional Complex_CoT / Response generation
"""

from cot_pipeline_accelerated import main


if __name__ == "__main__":
    main()
