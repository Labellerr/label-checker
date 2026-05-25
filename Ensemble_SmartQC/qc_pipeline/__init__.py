"""
qc_pipeline — SmartQC integration with label-checker.

Import the factory to get any validator:

    from qc_pipeline.validators import create_validator
    validator = create_validator("smartqc")
    validator = create_validator("gemini", api_key="...", model_name="gemini-2.5-flash")

Or use the full workflow:

    from qc_pipeline.smartqc_workflow import SmartQCWorkflow
    workflow = SmartQCWorkflow(output_dir="output")
    state, results, summary = workflow.run(
        coco_json_path="annotations.json",
        images_dir="images/"
    )
"""
