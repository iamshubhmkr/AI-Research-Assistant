"""
AWS CDK stack — full production infrastructure.
Deploy: cdk deploy
"""
import aws_cdk as cdk
from aws_cdk import (
    aws_bedrock as bedrock, aws_rds as rds, aws_ec2 as ec2,
    aws_dynamodb as dynamodb, aws_s3 as s3,
    aws_elasticache as elasticache, aws_cloudwatch as cw,
)
from constructs import Construct


class ResearchAssistantStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ElastiCache Redis — all 4 cache layers share one cluster
        elasticache.CfnCacheCluster(self, "Cache",
            cache_node_type="cache.t3.medium",
            engine="redis", num_cache_nodes=1)

        # PostgreSQL — LangGraph checkpoints (never auto-delete)
        rds.DatabaseInstance(self, "CheckpointDB",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16),
            instance_type=ec2.InstanceType("t3.small"),
            removal_policy=cdk.RemovalPolicy.RETAIN)

        # DynamoDB — sessions (pay-per-request = scales to zero)
        dynamodb.Table(self, "Sessions",
            partition_key=dynamodb.Attribute(
                name="session_id", type=dynamodb.AttributeType.STRING),
            time_to_live_attribute="ttl",
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST)

        # S3 — papers + golden dataset (versioned for safety)
        s3.Bucket(self, "Papers", versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=cdk.RemovalPolicy.RETAIN)

        # Bedrock Guardrails — content + grounding policy
        bedrock.CfnGuardrail(self, "Guardrail",
            name="research-assistant-guardrail",
            blocked_input_messaging="Request blocked by content policy.",
            blocked_outputs_messaging="Response blocked by content policy.",
            content_policy_config={"filtersConfig": [
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            ]},
            grounding_policy_config={"filtersConfig": [
                {"type": "GROUNDING", "threshold": 0.7}
            ]})

        # CloudWatch dashboard
        dash = cw.Dashboard(self, "Dash", dashboard_name="ResearchAssistant")
        ns   = "ResearchAssistant"
        dash.add_widgets(
            cw.GraphWidget(title="Latency P50/P95", left=[
                cw.Metric(namespace=ns, metric_name="QueryLatencyMs", statistic="p50"),
                cw.Metric(namespace=ns, metric_name="QueryLatencyMs", statistic="p95"),
            ]),
            cw.GraphWidget(title="RAGAS Faithfulness", left=[
                cw.Metric(namespace=ns, metric_name="RAGASFaithfulness"),
            ]),
            cw.GraphWidget(title="Cache Hit Rates", left=[
                cw.Metric(namespace=ns, metric_name="L1CacheHitRate"),
                cw.Metric(namespace=ns, metric_name="L2CacheHitRate"),
            ]),
        )

        # CloudWatch alarms
        cw.Alarm(self, "LatencyAlarm",
            metric=cw.Metric(namespace=ns, metric_name="QueryLatencyMs", statistic="p95"),
            threshold=15000, evaluation_periods=2,
            alarm_description="P95 latency > 15s")

        cw.Alarm(self, "FaithfulnessAlarm",
            metric=cw.Metric(namespace=ns, metric_name="RAGASFaithfulness"),
            threshold=0.75, comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=3, alarm_description="RAGAS faithfulness dropped below 0.75")
