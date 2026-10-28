import pandas as pd
import json
from pathlib import Path
import importlib.util, sys

# 1) Import de ta classe depuis le fichier fourni
spec = importlib.util.spec_from_file_location("IntelligentModeling", "IntelligentModeling.py")
IM = importlib.util.module_from_spec(spec)
sys.modules["IntelligentModeling"] = IM
spec.loader.exec_module(IM)
IntelligentModeling = IM.IntelligentModeling

# 2) Ajout des méthodes manquantes: load_dataset() et create_profile()
def _load_dataset(self, path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Format '{ext}' non supporté. Utilisez CSV, JSON ou Excel.")

def _create_profile(self) -> str:
    ctx = self.contextualisation
    res = self.results
    lines = []
    lines.append("# Profil du dataset\n")
    lines.append(f"**Use Cases** : `{ctx.get('use_cases', 'N/A')}`")
    lines.append(f"**Cible** : `{ctx.get('target')}`")
    lines.append(f"**Contexte global** : `{ctx.get('global_conxtexts', 'N/A')}`")
    shape = res.get('dataset_shape')
    if shape:
        lines.append(f"**Shape** : {shape[0]} lignes × {shape[1]} colonnes")

    mv = res.get('missing_values', {})
    crit = mv.get('critical_columns', [])
    lines.append("\n## Valeurs manquantes")
    lines.append(f"- Colonnes critiques (>50% NA) : {', '.join(crit) if crit else 'Aucune'}")

    ta = res.get('target_analysis', {})
    lines.append("\n## Analyse de la cible")
    lines.append(f"- Colonne: `{ta.get('target_column')}`  \n- Valeurs uniques: {ta.get('unique_values')}")
    if 'class_distribution' in ta:
        lines.append("- Distribution des classes:")
        for k, v in ta['class_distribution'].items():
            lines.append(f"  - {k}: {round(v, 3)}")
        lines.append(f"- Ratio de déséquilibre: {round(ta.get('imbalance_ratio', 0), 3)}")
    elif 'target_stats' in ta:
        lines.append("- Statistiques (régression):")
        for k, v in ta['target_stats'].items():
            try:
                vv = round(float(v), 4)
            except Exception:
                vv = v
            lines.append(f"  - {k}: {vv}")

    md = "\n".join(lines)
    #Path("profile.md").write_text(md, encoding="utf-8")
    Path("metadata.json").write_text(json.dumps({"contextualisation": ctx, "results": res}, indent=2), encoding="utf-8")
    return md

setattr(IntelligentModeling, "load_dataset", _load_dataset)
setattr(IntelligentModeling, "create_profile", _create_profile)

# 3) Création d’un mini dataset d’exemple (RPE -> régression)
df = pd.DataFrame({
    "athlete_id": [1, 2, 3, 4, 5, 6],
    "session_duration_min": [45, 60, None, 50, None, 40],
    "avg_heart_rate": [140, 155, None, 148, 162, 135],
    "sleep_hours_prev_night": [7.0, 6.5, 8.0, 6.0, 5.5, 7.5],
    "RPE": [5.0, 6.5, 4.0, 5.5, 7.0, None],  # cible
})
csv_path = "sample_rpe.csv"
df.to_csv(csv_path, index=False)

# 4) Exécution de l’analyse
analyser = IntelligentModeling()
analyser.contextualisation["use_cases"] = "Prediction RPE"
analyser.contextualisation["global_conxtexts"] = "Regression"   # on force le contexte global ici
profile_md = analyser.run_full_analysis(csv_path, target="RPE")

# 5) Affichage + fichiers
print(profile_md[:2000])
print("\nFichiers générés :")
print("- sample_rpe.csv")
print("- profile.md")
print("- metadata.json")
