from fipy.ngsi.headers import FiwareContext
from fipy.ngsi.orion import OrionClient
from fipy.wait import wait_for_orion
import json
from typing import List, Optional
from uri import URI


TENANT = 'ai4sdw'
ORION_EXTERNAL_BASE_URL = 'http://localhost:1026'
ORION_EXTERNAL_BASE_URL_CLUSTER = 'http://10.140.106.105/orion'
AI4SDW_INTERNAL_BASE_URL = 'http://ai4sdw:8082'
AI4SDW_INTERNAL_BASE_URL_CLUSTER = 'http://10.152.183.133:8000'
QUANTUMLEAP_INTERNAL_BASE_URL = 'http://quantumleap:8668'
QUANTUMLEAP_INTERNAL_BASE_URL_CLUSTER = 'http://10.152.183.129:8668'
AI4SDW_SUB = {
    "description": "Notify ai4sdw of changes to any entity.",
    "subject": {
        "entities": [
            {
                "idPattern": ".*"
            }
        ]
    },
    "notification": {
        "http": {
            "url": f"{AI4SDW_INTERNAL_BASE_URL}/updates"
        }
    }
}
AI4SDW_SUB_CLUSTER = {
    "description": "Notify ai4sdw of changes to any entity.",
    "subject": {
        "entities": [
            {
                "idPattern": ".*"
            }
        ]
    },
    "notification": {
        "http": {
            "url": f"{AI4SDW_INTERNAL_BASE_URL_CLUSTER}/updates"
        }
    }
}
QUANTUMLEAP_SUB = {
    "description": "Notify QuantumLeap of changes to any entity.",
    "subject": {
        "entities": [
            {
                "idPattern": ".*"
            }
        ]
    },
    "notification": {
        "http": {
            "url": f"{QUANTUMLEAP_INTERNAL_BASE_URL}/v2/notify"
        }
    }
}
QUANTUMLEAP_SUB_CLUSTER = {
    "description": "Notify QuantumLeap of changes to any entity.",
    "subject": {
        "entities": [
            {
                "idPattern": ".*"
            }
        ]
    },
    "notification": {
        "http": {
            "url": f"{QUANTUMLEAP_INTERNAL_BASE_URL_CLUSTER}/v2/notify"
        }
    }
}


def orion_client(env,
        service_path: Optional[str] = None,
                 correlator: Optional[str] = None) -> OrionClient:
    if env == "docker":
        base_url = URI(ORION_EXTERNAL_BASE_URL)
    elif env == "cluster":
        base_url = URI(ORION_EXTERNAL_BASE_URL_CLUSTER)
    ctx = FiwareContext(service=TENANT, service_path=service_path,
                        correlator=correlator)
    return OrionClient(base_url, ctx)


def wait_on_orion(env):
    wait_for_orion(orion_client(env))


class SubMan:

    def __init__(self, env):
        self.env = env
        self._orion = orion_client(self.env)

    def create_subscriptions(self) -> List[dict]:
        if self.env == "docker":
            self._orion.subscribe(AI4SDW_SUB)
            self._orion.subscribe(QUANTUMLEAP_SUB)
        elif self.env == "cluster":
            self._orion.subscribe(AI4SDW_SUB_CLUSTER)
            self._orion.subscribe(QUANTUMLEAP_SUB_CLUSTER)
        return self._orion.list_subscriptions()

# NOTE. Subscriptions and FIWARE service path.
# The way it behaves for subscriptions is a bit counter intuitive.
# You'd expect that with a header of 'fiware-servicepath: /' Orion would
# notify you of changes to *any* entities in the tree, similar to queries.
# But in actual fact, to do that you'd have to omit the service path header,
# which is what we do here. Basically the way it works is that if you
# specify a service path, then Orion only considers entities right under
# the last node in the service path, but not any other entities that might
# sit further down below. E.g. if your service tree looks like (e stands
# for entity)
#
#                        /
#                     p     q
#                  e1   r     e4
#                     e2 e3
#
# then a subscription with a service path of '/' won't catch any entities
# at all whereas one with a service path of '/p' will consider changes to
# e1 but not e2 nor e3.


def create_subscriptions(env):
    print(
        f"Creating catch-all {TENANT} entities subscription for QuantumLeap.")
    print(
        f"Creating catch-all {TENANT} entities subscription for AI4SDW.")

    man = SubMan(env)
    orion_subs = man.create_subscriptions()
    formatted = json.dumps(orion_subs, indent=4)

    print("Current subscriptions in Orion:")
    print(formatted)

