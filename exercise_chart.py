"""
NutriForge AI — Exercise Plan API  v3.0
Generates personalised weekly gym exercise plans using exercise_data.csv.
No external AI dependency — plans are built deterministically via weighted exercise scoring.

Deploy: uvicorn exercise_chart:app --host 0.0.0.0 --port $PORT
"""

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exercise_chart")


# ─── LOAD EXERCISE CSV ────────────────────────────────────────────────────────
def _locate_csv() -> Path:
    candidates = [
        Path(__file__).parent / "exercise_data.csv",
        Path("exercise_data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"exercise_data.csv not found. Tried: {[str(c) for c in candidates]}"
    )


ex_df: Optional[pd.DataFrame] = None
EX_LOAD_ERROR: Optional[str] = None

try:
    _csv_path = _locate_csv()
    _raw = pd.read_csv(_csv_path)

    required_columns = {
        "exercise_id", "name", "type", "equipment", "equipment_db_key",
        "muscle", "difficulty", "sets", "reps", "duration_min",
        "calories_burned_per_set", "rest_sec", "notes", "goal_tags", "body_part_tag",
    }
    missing_cols = required_columns - set(_raw.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    if _raw.empty:
        raise ValueError("exercise_data.csv is empty.")

    _raw["goal_tags"] = _raw["goal_tags"].fillna("").apply(
        lambda x: [i.strip().lower() for i in str(x).split(";") if i.strip()]
    )
    _raw["body_part_tag"] = _raw["body_part_tag"].fillna("").apply(
        lambda x: [i.strip().lower() for i in str(x).split(";") if i.strip()]
    )
    _raw["equipment_db_key"] = _raw["equipment_db_key"].fillna("").str.lower().str.strip()
    _raw["type"]             = _raw["type"].fillna("").str.lower().str.strip()
    _raw["difficulty"]       = _raw["difficulty"].fillna("beginner").str.lower().str.strip()
    _raw["muscle"]           = _raw["muscle"].fillna("").str.lower().str.strip()
    _raw["name"]             = _raw["name"].fillna("").str.strip()

    ex_df = _raw
    logger.info(f"[ExerciseChart] Loaded {len(ex_df)} exercises from {_csv_path}")

except Exception as _e:
    EX_LOAD_ERROR = str(_e)
    logger.error(f"[ExerciseChart] CSV load failed: {_e}")


def require_data() -> None:
    if ex_df is None or ex_df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Exercise dataset unavailable: {EX_LOAD_ERROR}",
        )


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ALWAYS_ALLOWED_EQUIPMENT = {"bodyweight"}

EXPERIENCE_TO_DIFFICULTY: dict[str, list[str]] = {
    "beginner":     ["beginner"],
    "intermediate": ["beginner", "intermediate"],
    "advanced":     ["beginner", "intermediate", "advanced"],
}

GOAL_PRIORITY: dict[str, list[str]] = {
    "muscle_gain": ["strength"],
    "fat_loss":    ["cardio", "strength"],
    "maintenance": ["strength", "cardio"],
    "strength":    ["strength"],
    "endurance":   ["cardio", "strength"],
}

VALID_GOALS       = set(GOAL_PRIORITY.keys())
VALID_EXPERIENCES = set(EXPERIENCE_TO_DIFFICULTY.keys())
VALID_SPLITS      = {"auto", "push_pull_legs", "upper_lower", "full_body", "bro_split"}

VOLUME_BY_HOURS: dict[float, tuple[int, int]] = {
    0.5: (2, 3),
    1.0: (4, 5),
    1.5: (6, 7),
    2.0: (8, 10),
    3.0: (10, 12),
}

# Split muscle-group assignments
SPLIT_FOCUS: dict[str, list[str]] = {
    "push_pull_legs": ["chest,shoulders,triceps", "back,biceps", "legs,glutes",
                       "chest,shoulders,triceps", "back,biceps", "legs,glutes", "rest"],
    "upper_lower":    ["upper", "lower", "upper", "lower", "upper", "lower", "rest"],
    "full_body":      ["full body"] * 6 + ["rest"],
    "bro_split":      ["chest", "back", "legs", "shoulders", "arms", "core", "rest"],
}

DAYS        = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_SHORT  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def volume_range(hours: float) -> tuple[int, int]:
    for k in sorted(VOLUME_BY_HOURS.keys()):
        if hours <= k:
            return VOLUME_BY_HOURS[k]
    return VOLUME_BY_HOURS[3.0]


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────
class ExercisePlanRequest(BaseModel):
    name:                str
    age:                 int             = Field(..., ge=10, le=80)
    gender:              str             = Field(..., description="'male' or 'female'")
    weight_kg:           Optional[float] = None
    height_cm:           Optional[float] = None
    goal:                str             = Field(..., description="muscle_gain | fat_loss | maintenance | strength | endurance")
    experience:          str             = Field(..., description="beginner | intermediate | advanced")
    gym_days:            int             = Field(..., ge=1, le=7)
    gym_hours:           float           = Field(..., ge=0.5, le=3.0)
    split:               Optional[str]   = Field("auto", description="auto | push_pull_legs | upper_lower | full_body | bro_split")
    injuries:            Optional[str]   = ""
    available_equipment: list[str]       = Field(
        default=[],
        description=(
            "List of equipment_db_key values from Firestore `equipment` collection. "
            "Only strength exercises matching these keys are included. "
            "Cardio and bodyweight are always included."
        ),
    )
    exercise_type:       Optional[str]   = Field(
        default=None,
        description="'cardio' | 'strength' | None (None = both cardio+strength). Priority over goal.",
    )
    power_id:            Optional[str]   = ""
    phone:               Optional[str]   = ""

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_GOALS:
            raise ValueError(f"goal must be one of {sorted(VALID_GOALS)}")
        return v

    @field_validator("experience")
    @classmethod
    def validate_experience(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_EXPERIENCES:
            raise ValueError(f"experience must be one of {sorted(VALID_EXPERIENCES)}")
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: Optional[str]) -> str:
        if v is None:
            return "auto"
        v = v.lower().strip()
        if v not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"male", "female", "other"}:
            raise ValueError("gender must be 'male', 'female', or 'other'")
        return v


class QuickFilterRequest(BaseModel):
    goal:                Optional[str]   = None
    experience:          Optional[str]   = "beginner"
    exercise_type:       Optional[str]   = None
    muscle:              Optional[str]   = None
    available_equipment: list[str]       = []
    limit:               int             = Field(20, ge=1, le=100)


# ─── CORE FILTER ──────────────────────────────────────────────────────────────
def filter_exercises(
    available_equipment: list[str],
    goal:          Optional[str] = None,
    experience:    Optional[str] = None,
    exercise_type: Optional[str] = None,
    muscle:        Optional[str] = None,
) -> pd.DataFrame:
    """
    Priority: exercise_type > goal > equipment

    exercise_type == "cardio"   → only cardio rows (equipment irrelevant)
    exercise_type == "strength" → only strength rows that match available_equipment;
                                  bodyweight used as fallback when no equipment match
    exercise_type == None       → cardio + strength (combined); strength filtered by equipment
    """
    require_data()

    norm_equip = {e.lower().strip() for e in available_equipment}
    et = (exercise_type or "").lower().strip()

    is_bodyweight = ex_df["equipment_db_key"].isin(ALWAYS_ALLOWED_EQUIPMENT)
    is_equipped   = ex_df["equipment_db_key"].isin(norm_equip)

    if et == "cardio":
        # Cardio also respects equipment — only cardio exercises whose equipment is
        # available (or bodyweight/no-equipment cardio)
        is_cardio = ex_df["type"] == "cardio"
        if norm_equip:
            result = ex_df[is_cardio & (is_bodyweight | is_equipped)].copy()
            if result.empty:
                # fallback: bodyweight cardio only
                result = ex_df[is_cardio & is_bodyweight].copy()
        else:
            result = ex_df[is_cardio & is_bodyweight].copy()

    elif et == "strength":
        # Strength must match available equipment; bodyweight-only fallback
        is_strength = ex_df["type"] == "strength"
        matched = ex_df[is_strength & is_equipped].copy()
        if matched.empty:
            result = ex_df[is_strength & is_bodyweight].copy()
        else:
            bw_strength = ex_df[is_strength & is_bodyweight]
            result = pd.concat([matched, bw_strength]).drop_duplicates("exercise_id")

    else:
        # Combined — cardio filtered by equipment, strength filtered by equipment
        is_cardio   = ex_df["type"] == "cardio"
        is_strength = ex_df["type"] == "strength"
        if norm_equip:
            cardio_pool   = ex_df[is_cardio   & (is_bodyweight | is_equipped)]
            strength_pool = ex_df[is_strength & (is_bodyweight | is_equipped)]
        else:
            cardio_pool   = ex_df[is_cardio   & is_bodyweight]
            strength_pool = ex_df[is_strength & is_bodyweight]
        result = pd.concat([cardio_pool, strength_pool]).drop_duplicates("exercise_id").copy()

    if goal:
        g = goal.lower()
        result = result[result["goal_tags"].apply(lambda gt: g in gt or "all" in gt)]
    if experience:
        allowed_diff = EXPERIENCE_TO_DIFFICULTY.get(experience.lower(), ["beginner"])
        result = result[result["difficulty"].isin(allowed_diff)]
    if muscle:
        m = muscle.lower()
        result = result[result["muscle"].str.contains(m, na=False)]

    return result.reset_index(drop=True)


# ─── CATALOGUE BUILDER ────────────────────────────────────────────────────────
def _row_to_dict(r: pd.Series) -> dict:
    return {
        "name":        r["name"],
        "muscle":      r["muscle"],
        "difficulty":  r["difficulty"],
        "sets":        str(r["sets"])         if pd.notna(r.get("sets"))         else None,
        "reps":        str(r["reps"])         if pd.notna(r.get("reps"))         else None,
        "duration":    str(r["duration_min"]) if pd.notna(r.get("duration_min")) else None,
        "equipment":   r["equipment"],
        "rest_sec":    int(r["rest_sec"])                        if pd.notna(r.get("rest_sec"))                        else 60,
        "cal_per_set": int(r["calories_burned_per_set"])         if pd.notna(r.get("calories_burned_per_set"))         else 0,
        "notes":       r["notes"]             if pd.notna(r.get("notes"))         else "",
        "goal_tags":   r["goal_tags"],
        "body_part_tag": r["body_part_tag"],
    }


def build_exercise_catalogue(
    available_equipment: list[str],
    goal:          str,
    experience:    str,
    exercise_type: Optional[str] = None,
) -> dict:
    df_filtered = filter_exercises(
        available_equipment,
        goal=goal,
        experience=experience,
        exercise_type=exercise_type,
    )

    cardio_df   = df_filtered[df_filtered["type"] == "cardio"]
    strength_df = df_filtered[df_filtered["type"] == "strength"]

    cardio_list   = [_row_to_dict(pd.Series(r._asdict())) for r in cardio_df.itertuples(index=False)]
    strength_list = [_row_to_dict(pd.Series(r._asdict())) for r in strength_df.itertuples(index=False)]

    return {
        "cardio":        cardio_list,
        "strength":      strength_list,
        "exercise_type": exercise_type,   # propagate so generate_plan can use it
        "total":         len(df_filtered),
    }


# ─── EXERCISE SCORING (NutriForge-style weighted selection) ───────────────────
def _score_exercise(ex: dict, goal: str, experience: str, focus_muscles: list[str]) -> float:
    score = 1.0

    # Goal alignment — dominates selection
    goal_tags = ex.get("goal_tags") or []
    if goal in goal_tags:
        score += 3.0
    elif "all" in goal_tags:
        score += 0.5
    else:
        score -= 0.5  # penalise exercises not matching this goal

    # Muscle focus — exact match preferred, partial match secondary
    ex_muscle = (ex.get("muscle") or "").lower()
    exact_match = any(m == ex_muscle for m in focus_muscles if m and m != "rest")
    partial_match = (not exact_match) and any(m and m in ex_muscle for m in focus_muscles if m != "rest")
    if exact_match:
        score += 2.5
    elif partial_match:
        score += 1.0

    # Difficulty preference
    diff_bonus = {
        "beginner":     {"beginner": 1.0, "intermediate": 0.3, "advanced": 0.0},
        "intermediate": {"beginner": 0.5, "intermediate": 1.0, "advanced": 0.5},
        "advanced":     {"beginner": 0.2, "intermediate": 0.7, "advanced": 1.2},
    }
    ex_diff = (ex.get("difficulty") or "beginner").lower()
    score += diff_bonus.get(experience, {}).get(ex_diff, 0.5)

    # Calorie burn bonus (normalised)
    cal = ex.get("cal_per_set") or 0
    score += min(cal / 50.0, 1.0)

    return score


def _resolve_split(req: ExercisePlanRequest) -> str:
    if req.split != "auto":
        return req.split
    if req.gym_days <= 3:
        return "full_body"
    if req.gym_days == 4:
        return "upper_lower"
    return "push_pull_legs"


def _day_focus_muscles(split: str, day_index: int) -> list[str]:
    focuses = SPLIT_FOCUS.get(split, ["full body"] * 7)
    focus_str = focuses[day_index % len(focuses)]
    return [m.strip().lower() for m in focus_str.split(",")]


# Day-type for combined mode cycling
_COMBINED_DAY_CYCLE = ["cardio", "strength", "mixed", "cardio", "strength", "mixed", "rest"]


def _pick_exercises(
    catalogue: dict,
    goal: str,
    experience: str,
    focus_muscles: list[str],
    target: int,
    injuries: str,
    used_names: set,
    day_type: Optional[str] = None,   # "cardio" | "strength" | "mixed" | None
) -> list[dict]:
    """
    day_type controls which pool to draw from:
      "cardio"   → only cardio pool
      "strength" → only strength pool
      "mixed"    → both pools, interleaved
      None       → respects GOAL_PRIORITY (legacy behaviour)
    """
    injury_keywords = [w.strip().lower() for w in (injuries or "").split(",") if w.strip()]

    if day_type == "cardio":
        pool = list(catalogue.get("cardio", []))
    elif day_type == "strength":
        pool = list(catalogue.get("strength", []))
    elif day_type == "mixed":
        # 50/50 split: score + pick strength and cardio independently then combine
        s_pool = list(catalogue.get("strength", []))
        c_pool = list(catalogue.get("cardio", []))
        s_count = target // 2
        c_count = target - s_count

        def _score_pool(p: list) -> list:
            scored_p = [(e, _score_exercise(e, goal, experience, focus_muscles)) for e in p]
            scored_p = [(e, sc * (0.3 if e["name"] in used_names else 1.0)) for e, sc in scored_p]
            scored_p.sort(key=lambda x: x[1] + random.uniform(0, 0.15), reverse=True)
            return [e for e, _ in scored_p]

        s_sorted = _score_pool(s_pool)
        c_sorted = _score_pool(c_pool)

        seen_mix: set = set()
        picked_s, picked_c = [], []
        for e in s_sorted:
            if len(picked_s) >= s_count: break
            if e["name"] not in seen_mix:
                picked_s.append(e); seen_mix.add(e["name"])
        for e in c_sorted:
            if len(picked_c) >= c_count: break
            if e["name"] not in seen_mix:
                picked_c.append(e); seen_mix.add(e["name"])

        # interleave: strength, cardio, strength, cardio …
        pool = []
        for pair in zip(picked_s, picked_c):
            pool.extend(pair)
        pool.extend(picked_s[len(picked_c):])
        pool.extend(picked_c[len(picked_s):])
        # return early — already selected
        used_names.update(e["name"] for e in pool)
        return pool
    else:
        goal_types = GOAL_PRIORITY.get(goal, ["strength", "cardio"])
        pool = []
        for t in goal_types:
            pool.extend(catalogue.get(t, []))

    # Filter injured muscles
    if injury_keywords:
        pool = [
            e for e in pool
            if not any(kw in (e.get("muscle") or "").lower() for kw in injury_keywords)
        ]

    # Score — goal + exact muscle match dominate
    scored = [(e, _score_exercise(e, goal, experience, focus_muscles)) for e in pool]

    # Penalise recently used
    scored = [(e, s * (0.3 if e["name"] in used_names else 1.0)) for e, s in scored]

    # Sort descending with small shuffle to prevent identical consecutive plans
    scored.sort(key=lambda x: x[1] + random.uniform(0, 0.15), reverse=True)

    selected = []
    seen: set = set()
    for ex, _ in scored:
        if len(selected) >= target:
            break
        if ex["name"] not in seen:
            selected.append(ex)
            seen.add(ex["name"])

    return selected


def _format_exercise(ex: dict) -> dict:
    return {
        "name": ex["name"],
        "sets": int(ex["sets"]) if ex.get("sets") and str(ex["sets"]).isdigit() else 3,
        "reps": ex.get("reps") or "10-12",
        "rest": f"{ex.get('rest_sec', 60)}s",
        "tip":  (ex.get("notes") or "Maintain good form.")[:80],
    }


# ─── PLAN GENERATOR ───────────────────────────────────────────────────────────
def generate_plan(req: ExercisePlanRequest, catalogue: dict) -> dict:
    vol_min, vol_max = volume_range(req.gym_hours)
    target    = (vol_min + vol_max) // 2
    split     = _resolve_split(req)
    et        = (req.exercise_type or "").lower().strip()   # "cardio" | "strength" | ""
    is_combined = not et  # no exercise_type = combined mode

    schedule   = []
    used_names: set = set()
    active_day_count = 0  # counts only gym days for cycle indexing

    for i, (day, short) in enumerate(zip(DAYS, DAYS_SHORT)):
        is_active = i < req.gym_days
        focus_muscles = _day_focus_muscles(split, i)

        if is_active:
            # Determine day_type:
            # - cardio only → always "cardio"
            # - strength only → always "strength"
            # - combined → cycle: cardio / strength / mixed
            if et == "cardio":
                day_type    = "cardio"
                focus_label = "Cardio"
            elif et == "strength":
                day_type    = "strength"
                focus_label = " & ".join(m.title() for m in focus_muscles if m not in ("rest",))
            else:
                # Combined: cycle through cardio → strength → mixed
                cycle_pos   = active_day_count % 3
                day_type    = ["cardio", "strength", "mixed"][cycle_pos]
                focus_label = {
                    "cardio":   "Cardio",
                    "strength": " & ".join(m.title() for m in focus_muscles if m not in ("rest",)),
                    "mixed":    "Cardio + Strength",
                }[day_type]

            exercises = _pick_exercises(
                catalogue, req.goal, req.experience,
                focus_muscles, target, req.injuries or "",
                used_names, day_type=day_type,
            )
            used_names.update(e["name"] for e in exercises)
            active_day_count += 1

            schedule.append({
                "day":       day,
                "day_short": short,
                "focus":     focus_label or req.goal.replace("_", " ").title(),
                "day_type":  day_type,
                "is_rest":   False,
                "exercises": [_format_exercise(e) for e in exercises],
            })
        else:
            schedule.append({
                "day":       day,
                "day_short": short,
                "focus":     "Rest & Recovery",
                "day_type":  "rest",
                "is_rest":   True,
                "exercises": [],
            })

    active_days = req.gym_days
    avg_ex = sum(len(d["exercises"]) for d in schedule if not d["is_rest"]) / max(active_days, 1)

    return {
        "split_type":            split,
        "exercise_mode":         et if et else "combined",
        "weekly_volume_sets":    int(avg_ex * active_days),
        "session_duration_mins": int(req.gym_hours * 60),
        "schedule":              schedule,
        "warmup":   ["5 min light walk", "Arm circles", "Leg swings", "Hip rotations"],
        "cooldown": ["5 min slow walk", "Quad stretch", "Hamstring stretch", "Deep breathing"],
    }


# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="NutriForge Exercise Plan API",
    description=(
        "Generates personalised weekly gym exercise plans using weighted exercise scoring.\n\n"
        "**Key constraint**: Strength exercises are filtered to only machines present "
        "in your Firestore `equipment` collection — pass `available_equipment` from the frontend.\n\n"
        "Cardio and bodyweight exercises are always available."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "ok",
        "service":   "NutriForge Exercise Plan API",
        "version":   "3.0.0",
        "endpoints": [
            "/exercise-plan",
            "/exercises",
            "/exercises/available",
            "/exercises/muscles",
            "/exercises/equipment-keys",
            "/health",
            "/docs",
        ],
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":          "running",
        "exercise_loaded": ex_df is not None,
        "total_exercises": len(ex_df) if ex_df is not None else 0,
        "load_error":      EX_LOAD_ERROR,
        "version":         "3.0.0",
    }


@app.post("/exercise-plan", tags=["Plan Generation"])
def generate_exercise_plan(req: ExercisePlanRequest):
    """
    Generate a full 7-day personalised exercise plan via weighted exercise scoring.

    **Frontend flow**:
    1. Fetch `equipment` collection from Firestore.
    2. Extract `equipment_db_key` values.
    3. Pass them as `available_equipment` in this request.
    """
    require_data()

    catalogue = build_exercise_catalogue(req.available_equipment, req.goal, req.experience, req.exercise_type)

    if catalogue["total"] == 0:
        catalogue = build_exercise_catalogue([], req.goal, req.experience, req.exercise_type)
        logger.warning(
            f"No exercises for equipment={req.available_equipment}. "
            "Fell back to bodyweight + cardio only."
        )

    if catalogue["total"] == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No exercises available for the given goal + experience combination. "
                "Check your exercise_data.csv content."
            ),
        )

    plan = generate_plan(req, catalogue)

    return JSONResponse(content={
        "member":                   req.name,
        "power_id":                 req.power_id,
        "goal":                     req.goal,
        "experience":               req.experience,
        "gym_days":                 req.gym_days,
        "gym_hours":                req.gym_hours,
        "available_exercise_count": catalogue["total"],
        "plan":                     plan,
    })


@app.get("/exercises", tags=["Exercise Data"])
def list_exercises(
    exercise_type: Optional[str] = None,
    muscle:        Optional[str] = None,
    difficulty:    Optional[str] = None,
    goal:          Optional[str] = None,
):
    """All exercises in the CSV (no equipment filter). Useful for admin/debug."""
    require_data()

    result = ex_df.copy()

    if exercise_type:
        result = result[result["type"] == exercise_type.lower()]
    if muscle:
        result = result[result["muscle"].str.contains(muscle.lower(), na=False)]
    if difficulty:
        result = result[result["difficulty"] == difficulty.lower()]
    if goal:
        g = goal.lower()
        result = result[result["goal_tags"].apply(lambda gt: g in gt)]

    return {
        "count":     len(result),
        "exercises": result.drop(columns=["goal_tags", "body_part_tag"]).to_dict("records"),
    }


@app.post("/exercises/available", tags=["Exercise Data"])
def available_exercises(req: QuickFilterRequest):
    """
    Returns exercises available given the gym's Firestore equipment list.
    Useful to preview what the plan generator will work with.
    """
    require_data()

    result = filter_exercises(
        req.available_equipment,
        goal=req.goal,
        experience=req.experience,
        exercise_type=req.exercise_type,
        muscle=req.muscle,
    )

    return {
        "count":     len(result),
        "filters":   req.dict(exclude={"limit"}),
        "exercises": result.drop(columns=["goal_tags", "body_part_tag"]).head(req.limit).to_dict("records"),
    }


@app.get("/exercises/muscles", tags=["Exercise Data"])
def muscle_groups():
    """All unique muscle groups in the dataset."""
    require_data()
    muscles = sorted(ex_df["muscle"].dropna().unique().tolist())
    return {"muscle_groups": muscles}


@app.get("/exercises/equipment-keys", tags=["Exercise Data"])
def equipment_keys():
    """
    All unique equipment_db_key values in the CSV.
    Store these as document IDs in your Firestore `equipment` collection
    and pass matching keys as `available_equipment` when calling /exercise-plan.
    """
    require_data()
    keys = sorted(ex_df["equipment_db_key"].dropna().unique().tolist())
    return {
        "equipment_keys": keys,
        "note": (
            "Store these as document IDs (or a field) in your Firestore `equipment` collection. "
            "Pass the matching keys in `available_equipment` when calling /exercise-plan."
        ),
    }
