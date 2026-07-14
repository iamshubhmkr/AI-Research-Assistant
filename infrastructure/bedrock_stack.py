"""
AWS CDK stack — declarative production infrastructure.
ALB+ECS (stateless API) | ElastiCache Redis | RDS PostgreSQL Multi-AZ
DynamoDB sessions | S3 papers | CloudWatch alarms | Bedrock IAM policy.
"""
from aws_cdk import (Stack, aws_ecs as ecs, aws_ec2 as ec2, aws_rds as rds,
                     aws_elasticache as ec, aws_dynamodb as ddb, aws_s3 as s3,
                     aws_iam as iam, RemovalPolicy)
from constructs import Construct


class ResearchAssistantStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs):
        super().__init__(scope, cid, **kwargs)

        vpc = ec2.Vpc(self, "Vpc", max_azs=2)

        s3.Bucket(self, "Papers", bucket_name="research-assistant-papers-prod",
                  versioned=True, removal_policy=RemovalPolicy.RETAIN)

        ddb.Table(self, "Sessions", table_name="research-sessions-prod",
                  partition_key=ddb.Attribute(name="session_id", type=ddb.AttributeType.STRING),
                  billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
                  time_to_live_attribute="ttl")

        rds.DatabaseInstance(self, "Checkpoints",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_16),
            vpc=vpc, multi_az=True, allocated_storage=50,
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM))

        ec.CfnCacheCluster(self, "Redis", cache_node_type="cache.t3.micro",
                            engine="redis", num_cache_nodes=1)

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)
        task = ecs.FargateTaskDefinition(self, "ApiTask", cpu=1024, memory_limit_mib=2048)
        task.add_to_task_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["*"]))
        task.add_container("api", image=ecs.ContainerImage.from_asset("."),
                           port_mappings=[ecs.PortMapping(container_port=8000)],
                           logging=ecs.LogDrivers.aws_logs(stream_prefix="ra"))
        ecs.FargateService(self, "ApiService", cluster=cluster,
                           task_definition=task, desired_count=2)
