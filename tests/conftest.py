import boto3
import pytest
from moto import mock_aws


@pytest.fixture
def table():
    """DynamoDB `vva-motion-jobs` table, mocked via moto.

    Shared across tests/langgraph_agents/test_motion_jobs.py, a worker test under
    tests/text-to-motion/, and further tests/langgraph_agents/ tests — hence the
    root-level conftest rather than a per-file fixture.
    """
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
