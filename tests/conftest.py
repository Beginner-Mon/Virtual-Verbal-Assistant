"""Root conftest.

NOTHING THIRD-PARTY MAY BE IMPORTED AT MODULE SCOPE HERE, pytest excepted.

pytest loads this file before collecting anything, for every suite under
tests/ — including suites installed from a completely different requirements
file. `import boto3` and `from moto import mock_aws` used to sit at the top,
and the SpeechLLm job (which installs SpeechLLm/requirements.txt, with no
boto3) died before running a single test:

    ImportError while loading conftest '.../tests/conftest.py'
    tests/conftest.py:1: in <module>
        import boto3
    E   ModuleNotFoundError: No module named 'boto3'
    Process completed with exit code 4

A module-scope import here is a dependency imposed on every job in the repo.
Imports belong inside the fixture that needs them, where only the tests that
actually request it pay for it. tests/langgraph_agents/
test_requirements_complete.py enforces this.
"""
import pytest


@pytest.fixture
def table():
    """DynamoDB `vva-motion-jobs` table, mocked via moto.

    Shared across tests/langgraph_agents/test_motion_jobs.py, a worker test under
    tests/text-to-motion/, and further tests/langgraph_agents/ tests — hence the
    root-level conftest rather than a per-file fixture.

    boto3 and moto are imported HERE, not at module scope: see this file's
    docstring. A suite that never requests `table` must not need them.
    """
    import boto3
    from moto import mock_aws

    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        t = ddb.create_table(
            TableName="vva-motion-jobs",
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "job_id", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "N"},
            ],
            GlobalSecondaryIndexes=[{
                "IndexName": "status-created_at-index",
                "KeySchema": [
                    {"AttributeName": "status", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }],
            BillingMode="PAY_PER_REQUEST",
        )
        yield t
