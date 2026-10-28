# aisport_knowledgebase.py / modules/KnowledgeBase.py
# Docker Command:
#   docker run -d --name weaviate -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
# OPENAI key in env:
#   export OPENAI_API_KEY="sk-xxxx"

import datetime
import json
import uuid
import glob
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.Arguments import arguments as arg
from modules.ContextSctack import ContextSctack

from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter


class KnowledgeBase:
    """
    AISport Knowledge Base:
    - Weaviate connection (cloud or local)
    - Schema with 6 named vectors:
        v_title, v_global_text, v_prompt,
        v_all, v_dataprofiling, v_pipeline
    - Ingestion from JSON metadata
    - Semantic search per vector + filters
    """

    DEFAULT_COLLECTION = "AISportUsageRecord"

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        weaviate_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        verbose: bool = True,
        create_if_missing: bool = True,
        drop_existing: bool = False,
    ):
        self.verbose = verbose
        self.collection_name = collection_name

        # 1) OpenAI key (we reuse arg.api_key)
        self.openai_api_key = arg.api_key

        # 2) Weaviate connection params
        self.weaviate_url = weaviate_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.weaviate_api_key = weaviate_api_key or os.getenv("WEAVIATE_API_KEY")
        self.client = self._connect()

        # 3) Schema
        if create_if_missing:
            self._ensure_schema(drop_existing=drop_existing)

        # 4) Collection handle
        self.collection = self.client.collections.get(self.collection_name)

    # ---------------------------
    # Setup & Connection
    # ---------------------------
    def _load_openai_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv()
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return key
        key_file = Path("OPENAI_API_KEY")
        if key_file.exists():
            text = key_file.read_text(encoding="utf-8").strip()
            if text:
                os.environ["OPENAI_API_KEY"] = text
                return text
        raise ValueError("OPENAI_API_KEY not found (env/.env/file).")

    def _connect(self):
        headers = {"X-OpenAI-Api-Key": self.openai_api_key}
        if self.weaviate_api_key:
            if self.verbose:
                print("🔗 Connecting to Weaviate Cloud…")
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(self.weaviate_api_key),
                headers=headers,
            )
        else:
            if self.verbose:
                print("🔗 Connecting to local Weaviate…")
            host = self.weaviate_url.split("://")[-1].split(":")[0]
            port = int(self.weaviate_url.split(":")[-1]) if ":" in self.weaviate_url else 8080
            client = weaviate.connect_to_local(host=host, port=port, headers=headers)

        if not client.is_ready():
            raise ConnectionError("Cannot connect to Weaviate.")
        if self.verbose:
            print("✅ Weaviate connection OK")
        return client

    def _ensure_schema(self, drop_existing: bool = False):
        """
        Schema aligned with generated JSON:
        - Properties: use_case, user_prompt, global_context, data_profiling, optimal_pipelines
        - 6 named vectors:
            v_title, v_global_text, v_prompt,
            v_all, v_dataprofiling, v_pipeline
        """
        exists = self.client.collections.exists(self.collection_name)
        if exists and drop_existing:
            if self.verbose:
                print(f"🗑️ Dropping existing collection '{self.collection_name}'")
            self.client.collections.delete(self.collection_name)
            exists = False

        if not exists:
            if self.verbose:
                print(f"🏗️ Creating collection '{self.collection_name}'…")
            self.client.collections.create(
                self.collection_name,
                properties=[
                    Property(name="use_case", data_type=DataType.TEXT, description="Use case label"),
                    Property(name="user_prompt", data_type=DataType.TEXT, description="User prompt"),
                    Property(name="global_context", data_type=DataType.TEXT, description="Global context label"),
                    Property(name="data_profiling", data_type=DataType.TEXT, description="Serialized data profiling"),
                    Property(name="optimal_pipelines", data_type=DataType.TEXT, description="Serialized optimal pipelines"),
                ],
                vector_config=[
                    # 1) title vector on use_case
                    Configure.Vectors.text2vec_openai(
                        name="v_title",
                        source_properties=["use_case"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                    # 2) global context
                    Configure.Vectors.text2vec_openai(
                        name="v_global_text",
                        source_properties=["global_context"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                    # 3) user prompt
                    Configure.Vectors.text2vec_openai(
                        name="v_prompt",
                        source_properties=["user_prompt"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                    # 4) v_all : global text (use_case + global_context + user_prompt)
                    Configure.Vectors.text2vec_openai(
                        name="v_all",
                        source_properties=["use_case", "global_context", "user_prompt"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                    # 5) v_dataprofiling : only on data_profiling
                    Configure.Vectors.text2vec_openai(
                        name="v_dataprofiling",
                        source_properties=["data_profiling"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                    # 6) v_pipeline : optimal_pipelines + context + profiling
                    Configure.Vectors.text2vec_openai(
                        name="v_pipeline",
                        source_properties=["optimal_pipelines", "global_context", "data_profiling"],
                        model="text-embedding-3-small",
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        ),
                    ),
                ],
            )

            if self.verbose:
                print("✅ Schema created")
        else:
            if self.verbose:
                print(f"ℹ️ Collection '{self.collection_name}' already exists (no recreation).")

    # ---------------------------
    # JSON ingestion
    # ---------------------------
    def extract_from_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remappe un JSON généré vers les propriétés du schéma.
        Gère proprement les cas où data_profiling / optimal_pipelines sont
        déjà sérialisés en string pour éviter le double encodage.
        """

        use_case = json_data.get("use_case", "")
        user_prompt = json_data.get("user_prompt", "")
        global_context = json_data.get("global_context", "")

        # --- data_profiling ---
        data_profiling = json_data.get("data_profiling", {})
        if isinstance(data_profiling, str):
            # c'est déjà du JSON sérialisé → on le garde tel quel
            dp_txt = data_profiling
        else:
            # dict / autre → on sérialise
            dp_txt = json.dumps(data_profiling, ensure_ascii=False)

        # --- optimal_pipelines ---
        optimal_pipelines = json_data.get("optimal_pipelines", {})

        # Si c'est une string, on essaie UNE fois de la décoder
        if isinstance(optimal_pipelines, str):
            try:
                tmp = json.loads(optimal_pipelines)
                optimal_pipelines = tmp
            except Exception:
                # ce n'est pas du JSON, on laisse tel quel
                pass

        # Maintenant si c'est un dict, on sérialise proprement
        if isinstance(optimal_pipelines, dict):
            op_txt = json.dumps(optimal_pipelines, ensure_ascii=False)
        else:
            # sinon on cast en string (au pire)
            op_txt = str(optimal_pipelines)

        return {
            "use_case": use_case,
            "user_prompt": user_prompt,
            "global_context": global_context,
            "data_profiling": dp_txt,
            "optimal_pipelines": op_txt,
        }

    def load_json_folder(self, folder_glob: str) -> List[Dict[str, Any]]:
        """
        Load and parse JSON files from a folder (glob pattern).
        """
        files = glob.glob(folder_glob)
        if not files:
            print(f"❌ No JSON files found for path: {folder_glob}")
            print("💡 Check the folder exists and pattern is correct (e.g. json_img_dataset2/*.json)")
            return []
        else:
            print(f"✅ {len(files)} files found in {folder_glob}")

        out, errors = [], []
        for path in sorted(files):
            if not path.lower().endswith(".json"):
                print(f"⚠️ Ignored non-JSON file: {path}")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out.append(self.extract_from_json(data))
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in {path}: {e}")
            except Exception as e:
                errors.append(f"{path}: {e}")

        if errors and self.verbose:
            print(f"⚠️ {len(errors)} ingestion errors (showing first 5):")
            for e in errors[:5]:
                print("  -", e)

        if self.verbose:
            print(f"✅ {len(out)} records parsed successfully")
        return out

    def batch_insert(
        self,
        records: List[Dict[str, Any]],
        batch_size: int = 64,
        pause_s: float = 0.25,
    ) -> Tuple[int, int]:
        ok, ko = 0, 0
        total = len(records)
        if self.verbose:
            print(f"🚀 Ingesting {total} objects (batch={batch_size})")
        for i in range(0, total, batch_size):
            batch = records[i: i + batch_size]
            try:
                with self.collection.batch.dynamic() as bx:
                    for rec in batch:
                        try:
                            bx.add_object(properties=rec, uuid=str(uuid.uuid4()))
                            ok += 1
                        except Exception as e:
                            if self.verbose:
                                print("  ❌ add_object:", e)
                            ko += 1
                if self.verbose:
                    print(f"  ✅ batch {(i // batch_size) + 1}/{(total + batch_size - 1) // batch_size}")
                if i + batch_size < total and pause_s:
                    time.sleep(pause_s)
            except Exception as e:
                if self.verbose:
                    print("  ❌ batch error:", e)
                ko += len(batch)
        if self.verbose:
            rate = 100 * ok / max(1, (ok + ko))
            print(f"🎯 Ingestion: OK={ok}, KO={ko}, success={rate:.1f}%")
        return ok, ko

    def count(self) -> int:
        agg = self.collection.aggregate.over_all(total_count=True)
        return agg.total_count

    # ---------------------------
    # Search
    # ---------------------------
    def search(
        self,
        query: str,
        vector: str = "v_all",   # default: global text
        limit: int = 5,
        filter_by: Optional[Dict[str, List[str]]] = None,
        return_props: Optional[List[str]] = None,
        with_distance: bool = True,
    ):
        filt = None
        if filter_by:
            for k, vals in filter_by.items():
                node = Filter.by_property(k).contains_any(vals)
                filt = node if filt is None else (filt & node)

        meta = MetadataQuery(distance=True) if with_distance else None
        res = self.collection.query.near_text(
            query=query,
            target_vector=vector,
            limit=limit,
            filters=filt,
            return_properties=return_props,
            return_metadata=meta,
        )
        out = []
        if res.objects:
            for obj in res.objects:
                properties = obj.properties or {}
                out.append(
                    {
                        "uuid": str(obj.uuid),
                        "distance": getattr(obj.metadata, "distance", None) if with_distance else None,
                        "properties": properties if return_props is None else {
                            k: properties.get(k) for k in return_props
                        },
                    }
                )
        return out

    def rerank_two_stage(
        self,
        query: str,
        stage1_vector: str = "v_title",
        stage1_limit: int = 10,
        stage2_vector: str = "v_all",
        stage2_limit: int = 5,
    ):
        """
        Two-stage:
          1) warmup on titles (v_title)
          2) scoring on stage2_vector with explicit properties
        """
        SAFE_PROPS = ["use_case", "user_prompt", "global_context", "data_profiling", "optimal_pipelines"]
        meta = MetadataQuery(distance=True)

        top_title_results = self.collection.query.near_text(
            query=query,
            target_vector=stage1_vector,
            limit=stage1_limit,
            return_metadata=meta,
        )

        res2 = None
        if top_title_results.objects:
            print("🔹 Phase 2: rerank among top titles on global context")
            uuids = [str(obj.uuid) for obj in top_title_results.objects]
            print(f"  UUIDs of top titles: {uuids}")

            res2 = self.collection.query.near_text(
                query=query,
                target_vector=stage2_vector,
                filters=Filter.by_id().contains_any(uuids),
                limit=stage2_limit,
                return_metadata=meta,
            )

        out = []
        for obj in (res2.objects or []):
            props = obj.properties or {}
            out.append(
                {
                    "uuid": str(obj.uuid),
                    "distance": getattr(obj.metadata, "distance", None),
                    "properties": {k: props.get(k) for k in SAFE_PROPS},
                }
            )
        return out

    # ---------------------------
    # Utilities
    # ---------------------------
    def fetch_examples(self, limit: int = 3):
        res = self.collection.query.fetch_objects(limit=limit)
        return [{"uuid": str(o.uuid), "properties": o.properties} for o in (res.objects or [])]

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    # ---------------------------
    # Regression pipelines
        # ---------------------------
    def get_best_regression_pipeline_with_mode(
        self,
        use_case: str,
        model_type: str | None = None,
        mode: str = "no_kb",
        query: str | None = None,
    ):
        if mode == "no_kb":
            return None, None

        # 1. Build a richer query: use the use_case and query combined
        search_query = query if query else use_case
        props = ["optimal_pipelines", "use_case", "global_context", "user_prompt"]

        # Mapping mode -> vector
        vector_name = "v_dataprofiling" if mode == "kb_high" else "v_all"

        # 2. RELAXED SEARCH: Removed the hard 'filter_by' to allow semantic similarity to work
        hits = self.search(
            query=search_query,
            vector=vector_name,
            limit=10, # Increased limit to find more candidates
            return_props=props,
            with_distance=True,
        )

        if not hits:
            return None, None

        best_payload = None
        best_sim = -1.0
        
        # Define thresholds
        # kb_low: very permissive (0.10)
        # kb_high: strict profiling match (0.25)
        threshold = 0.25 if mode == "kb_high" else 0.10

        for h in hits:
            dist = h["distance"]
            sim = 1.0 - dist if dist is not None else 0.0
            
            # Skip if similarity is too low for the chosen mode
            if sim < threshold:
                continue

            props_h = h.get("properties") or {}
            txt = props_h.get("optimal_pipelines")
            if not txt: continue
            
            # Robust Decoding
            try:
                payload = json.loads(txt) if isinstance(txt, str) else txt
                if isinstance(payload, str): payload = json.loads(payload) # Handle double encoding
                
                candidate = payload.get("best") if isinstance(payload, dict) and "best" in payload else payload
                
                if not isinstance(candidate, dict): continue

                # 3. SOFT MODEL FILTER: 
                # If we requested a specific model, try to match it. 
                # If no match found in the hit, we keep looking at other hits.
                if model_type and candidate.get("model_type") != model_type:
                    continue

                # If this hit is more relevant (higher similarity) than our previous best
                if sim > best_sim:
                    best_sim = sim
                    best_payload = candidate

            except Exception as e:
                print(f"⚠️ Decoding error in KB hit: {e}")
                continue

        # 4. FINAL FALLBACK: If we didn't find the specific model_type, 
        # try one more search WITHOUT the model_type constraint to get at least a valid pipeline
        if best_payload is None and model_type is not None:
             # Recursive call once without model constraint
             return self.get_best_regression_pipeline_with_mode(use_case, None, mode, query)

        return best_payload, (1.0 - best_sim, best_sim) if best_payload else (None, None)
    
    def store_best_regression_pipeline_to_file(
        self,
        pipeline_data: dict,
        context_stack: ContextSctack,
        folder: str = "datasets/json_img_dataset2/"
    ) -> str:
        os.makedirs(folder, exist_ok=True)

        use_case = pipeline_data.get("use_case") or getattr(context_stack, "use_case", "")
        dp = getattr(context_stack, "data_profiling", {})

        record = {
            "use_case": use_case,
            "user_prompt": f"Best pipeline for {use_case} | model={pipeline_data.get('model_type','?')}",
            "global_context": getattr(context_stack, "context_type", "Regression"),
            "data_profiling": json.dumps(dp, ensure_ascii=False),
            "optimal_pipelines": json.dumps({"best": pipeline_data}, ensure_ascii=False),
            "_meta": {
                "kind": "regression",
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "uuid": str(uuid.uuid4()),
                "model_type": pipeline_data.get("model_type"),
            },
        }

        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        short = record["_meta"]["uuid"].split("-")[0]
        safe_uc = "".join(c if c.isalnum() or c in "-_" else "_" for c in use_case)[:64]
        fname = f"{ts}_{safe_uc}_regression_{short}.json"
        fpath = os.path.join(folder, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"💾 Saved best regression pipeline → {fpath}")

        return fpath

    def store_best_regression_pipeline_in_kb(self, pipeline_data, context_stack):
        record = {
            "use_case": pipeline_data.get("use_case") or getattr(context_stack, "use_case", ""),
            "user_prompt": f"Best pipeline for {pipeline_data.get('use_case','?')} | model={pipeline_data.get('model_type','?')}",
            "global_context": getattr(context_stack, "context_type", "Regression"),
            "data_profiling": json.dumps(getattr(context_stack, "data_profiling", {}), ensure_ascii=False),
            "optimal_pipelines": json.dumps({"best": pipeline_data}, ensure_ascii=False),
        }
        self.batch_insert([record], batch_size=1)

    # ---------------------------
    # Classification pipelines
    # ---------------------------
    def get_best_classification_pipeline_with_mode(
        self,
        use_case: str,
        model_type: str | None = None,
        mode: str = "no_kb",
        query: str | None = None,
    ):
        if mode == "no_kb":
            return None, None

        # 1. Build a richer query: use the use_case and query combined
        search_query = query if query else use_case
        props = ["optimal_pipelines", "use_case", "global_context", "user_prompt"]

        # Mapping mode -> vector
        vector_name = "v_dataprofiling" if mode == "kb_high" else "v_all"

        # 2. RELAXED SEARCH: Removed the hard 'filter_by' to allow semantic similarity to work
        hits = self.search(
            query=search_query,
            vector=vector_name,
            limit=10, # Increased limit to find more candidates
            return_props=props,
            with_distance=True,
        )

        if not hits:
            return None, None

        best_payload = None
        best_sim = -1.0
        
        # Define thresholds
        # kb_low: very permissive (0.10)
        # kb_high: strict profiling match (0.25)
        threshold = 0.25 if mode == "kb_high" else 0.10

        for h in hits:
            dist = h["distance"]
            sim = 1.0 - dist if dist is not None else 0.0
            
            # Skip if similarity is too low for the chosen mode
            if sim < threshold:
                continue

            props_h = h.get("properties") or {}
            txt = props_h.get("optimal_pipelines")
            if not txt: continue
            
            # Robust Decoding
            try:
                payload = json.loads(txt) if isinstance(txt, str) else txt
                if isinstance(payload, str): payload = json.loads(payload) # Handle double encoding
                
                candidate = payload.get("best") if isinstance(payload, dict) and "best" in payload else payload
                
                if not isinstance(candidate, dict): continue

                # 3. SOFT MODEL FILTER: 
                # If we requested a specific model, try to match it. 
                # If no match found in the hit, we keep looking at other hits.
                if model_type and candidate.get("model_type") != model_type:
                    continue

                # If this hit is more relevant (higher similarity) than our previous best
                if sim > best_sim:
                    best_sim = sim
                    best_payload = candidate

            except Exception as e:
                print(f"⚠️ Decoding error in KB hit: {e}")
                continue

        # 4. FINAL FALLBACK: If we didn't find the specific model_type, 
        # try one more search WITHOUT the model_type constraint to get at least a valid pipeline
        if best_payload is None and model_type is not None:
             # Recursive call once without model constraint
             return self.get_best_classification_pipeline_with_mode(use_case, None, mode, query)

        return best_payload, (1.0 - best_sim, best_sim) if best_payload else (None, None)

    def store_best_classification_pipeline_in_kb(self, pipeline_data, context_stack):
        record = {
            "use_case": pipeline_data.get("use_case") or getattr(context_stack, "use_case", ""),
            "user_prompt": f"Best pipeline for {pipeline_data.get('use_case','?')} | model={pipeline_data.get('model_type','?')}",
            "global_context": getattr(context_stack, "context_type", "Classification"),
            "data_profiling": json.dumps(getattr(context_stack, "data_profiling", {}), ensure_ascii=False),
            "optimal_pipelines": json.dumps({"best": pipeline_data}, ensure_ascii=False),
        }
        self.batch_insert([record], batch_size=1)

    def store_best_classification_pipeline_to_file(
        self,
        pipeline_data: dict,
        context_stack: ContextSctack,
        folder: str = "datasets/json_img_dataset2/",
        user_prompt: str | None = None,
        global_context_override: str | None = None,
    ) -> str:
        os.makedirs(folder, exist_ok=True)

        use_case = pipeline_data.get("use_case") or getattr(context_stack, "use_case", "")
        user_prompt = user_prompt or getattr(context_stack, "user_prompt", "")
        gctx = global_context_override or getattr(context_stack, "global_context", "")

        dp = getattr(context_stack, "data_profiling", {})
        record = {
            "use_case": use_case,
            "user_prompt": user_prompt,
            "global_context": gctx,
            "data_profiling": json.dumps(dp, ensure_ascii=False),
            "optimal_pipelines": json.dumps({"best": pipeline_data}, ensure_ascii=False),
            "_meta": {
                "kind": "classification",
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "uuid": str(uuid.uuid4()),
                "model_type": pipeline_data.get("model_type"),
            },
        }

        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        short = record["_meta"]["uuid"].split("-")[0]
        safe_uc = "".join(c if c.isalnum() or c in "-_" else "_" for c in use_case)[:64]
        fname = f"{ts}_{safe_uc}_classification_{short}.json"
        fpath = os.path.join(folder, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"💾 Saved best classification pipeline → {fpath}")

        return fpath
