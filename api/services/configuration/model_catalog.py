"""Model configuration v2 catalog: the providers, models and Dograh-managed
options an organization can choose from.

Shared by `GET /organizations/model-configurations/v2/defaults` and the Agent
Copilot settings engine (`api/services/copilot/settings.py`). Pure (no DB, no
network): everything comes from the configuration registry.
"""

from __future__ import annotations

from typing import Any

from api.schemas.ai_model_configuration import (
    DOGRAH_DEFAULT_LANGUAGE,
    DOGRAH_DEFAULT_VOICE,
    DOGRAH_SPEED_MAX,
    DOGRAH_SPEED_MIN,
    DOGRAH_SPEED_OPTIONS,
    DOGRAH_SPEED_STEP,
)
from api.services.configuration.defaults import DEFAULT_SERVICE_PROVIDERS
from api.services.configuration.registry import (
    DOGRAH_MULTILINGUAL_AUTODETECT_LANGUAGES,
    DOGRAH_STT_LANGUAGES,
    REGISTRY,
    DograhTTSService,
    ServiceProviders,
    ServiceType,
)


def dograh_allows_custom_voice() -> bool:
    extra = DograhTTSService.model_fields["voice"].json_schema_extra
    if isinstance(extra, dict):
        return bool(extra.get("allow_custom_input", False))
    return False


def byok_provider_schemas(service_type: ServiceType) -> dict[str, dict]:
    """JSON schema per BYOK provider for one service type (Dograh excluded)."""
    return {
        provider: model_cls.model_json_schema()
        for provider, model_cls in REGISTRY[service_type].items()
        if provider != ServiceProviders.DOGRAH.value
    }


def get_model_configuration_v2_defaults() -> dict[str, Any]:
    """The v2 defaults/catalog payload (Dograh-managed options + BYOK schemas)."""
    byok_default_providers = {
        service: provider
        for service, provider in DEFAULT_SERVICE_PROVIDERS.items()
        if provider != ServiceProviders.DOGRAH.value
    }
    return {
        "dograh": {
            "voices": [DOGRAH_DEFAULT_VOICE],
            "allow_custom_input": dograh_allows_custom_voice(),
            "speeds": list(DOGRAH_SPEED_OPTIONS),
            "speed_range": {
                "min": DOGRAH_SPEED_MIN,
                "max": DOGRAH_SPEED_MAX,
                "step": DOGRAH_SPEED_STEP,
            },
            "languages": DOGRAH_STT_LANGUAGES,
            "multilingual_languages": DOGRAH_MULTILINGUAL_AUTODETECT_LANGUAGES,
            "defaults": {
                "voice": DOGRAH_DEFAULT_VOICE,
                "speed": 1.0,
                "language": DOGRAH_DEFAULT_LANGUAGE,
            },
        },
        "byok": {
            "pipeline": {
                "llm": byok_provider_schemas(ServiceType.LLM),
                "tts": byok_provider_schemas(ServiceType.TTS),
                "stt": byok_provider_schemas(ServiceType.STT),
                "embeddings": byok_provider_schemas(ServiceType.EMBEDDINGS),
                "default_providers": byok_default_providers,
            },
            "realtime": {
                "realtime": byok_provider_schemas(ServiceType.REALTIME),
                "llm": byok_provider_schemas(ServiceType.LLM),
                "embeddings": byok_provider_schemas(ServiceType.EMBEDDINGS),
                "default_providers": byok_default_providers,
            },
        },
    }
