"""
agents/automotive_agent.py — Single LLM-Led Pydantic AI Automotive Agent Architecture

Gemini classifies the request, selects zero or more safe tools, reasons over
their results, and produces one validated response. There is no Python intent
classifier, keyword router, fixed tool plan, or generated SQL.

Prefix Caching:
    AUTOMOTIVE_AGENT_PROMPT is sized to ~4,500 tokens — comfortably above the
    Gemini 3.x family caching threshold of 4,096 tokens. After the first request,
    Gemini serves the static prefix (Layer 1 + Layer 2) from the KV-cache at
    ~90% discount, dramatically reducing cost and TTFT on multi-turn sessions.

    Prompt structure (strictly ordered most-static → most-dynamic):
        Layer 1: Core policies         (~1,050 tokens) ← always cached
        Layer 2: Domain reference      (~3,450 tokens) ← always cached
        ─────────────────────────────────────────────── 4,096+ token threshold
        Dynamic suffix: history + user message         ← computed fresh
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional, Any, Literal

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models.google import GoogleModel as GeminiModel
from pydantic_ai.providers.google import GoogleProvider

from db.fuzzy_queries import fuzzy_engine
from db.queries import db_get_all_issues
from tools.car_tools import (
    get_all_cars,
    filter_cars_by_budget,
    filter_cars_by_fuel,
    filter_cars_by_segment,
    get_car_by_name,
    get_common_issue_info,
    get_service_intervals,
    calculate_emi,
)


@dataclass
class AutoBotDeps:
    """Request-scoped data passed safely to Pydantic AI tools."""
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    history_context: str = ""
    intent_labels: list[str] = field(default_factory=list)
    intent_confidence: dict[str, float] = field(default_factory=dict)
    tool_calls: list[str] = field(default_factory=list)
    user_profile: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AutoBotStreamUpdate:
    """A UI-safe update emitted after the agent graph processes an event."""
    content: str
    intents: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    complete: bool = False
    elapsed_seconds: float = 0.0
    cached_tokens: int = 0   # Tokens served from LLM prefix cache (0 = cache miss)


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMOTIVE_AGENT_PROMPT
#
# LAYER 1: Core Policies (~1,050 tokens) — unchanged from original
# LAYER 2: Indian Automotive Domain Reference Matrix (~3,450 tokens) — NEW
#
# Combined target: ~4,500 tokens. Gemini 3.x caching threshold: 4,096 tokens.
# This prompt MUST remain byte-for-byte identical across all requests to
# guarantee a cache hit on the prefix. Never inject dynamic values here.
# ─────────────────────────────────────────────────────────────────────────────

AUTOMOTIVE_AGENT_PROMPT = """You are AutoBot, one expert AI automobile
assistant for the Indian market. You handle buying, vehicle diagnostics,
maintenance, finance, and general automotive questions in one conversation.

## Your LLM intent-classification job
Before deciding on tools or drafting an answer, classify every explicit user
need into ALL applicable intents:
- buying: recommendations, comparisons, features, budget, purchase decisions
- diagnostics: symptoms, noises, warning lights, faults, drivability, safety
- service: maintenance, service intervals, parts replacement, service cost
- finance: EMI, loan, down payment, affordability, ownership cost
- general: all other automotive questions, including EV, insurance, rules,
  driving, accessories, and out-of-catalogue questions

This classification is your responsibility as the LLM. Never use or imply a
keyword router. As your first action for every request, call
`record_intent_classification` with every applicable intent. A user message can
have multiple intents.

## Tool-use policy
You have safe, read-only tools for local vehicle, issue, service, and finance
data. You decide which tools to use, their order, and their arguments.
1. Use a relevant tool whenever it can provide evidence for a claim.
2. For mixed requests, call every relevant tool; do not retrieve data only for
   the first intent.
3. Treat tool results as the source of truth. Never invent a catalogue model,
   specification, price, service interval, common issue, or calculation.
4. Tools are not a complete market database. For current prices, regulations,
   recalls, incentives, insurance terms, launches, or availability not returned
   by a tool, say live verified research is not configured and do not present
   your model memory as a current fact.
5. Never generate SQL, ask for credentials, or claim access to data a tool did
   not return.

## Automotive safety policy
Your diagnostic advice is triage, not a confirmed repair diagnosis. If the user
describes brake or steering loss, smoke/fire, fuel leak, severe overheating,
loss of control, or a critical warning, lead with a clear 'do not drive' and
professional-assistance recommendation. Do not provide risky repair steps.
Ask focused questions when the make, model, year, fuel type, warning code, or
driving conditions are needed for a reliable answer.

## Finance policy
Use the EMI calculator tool for every numeric EMI result. Do not do EMI math
yourself. State all material assumptions, including price basis, down payment,
rate, and tenure. A catalogue price is a base price, not an on-road quote.

## Response policy
After you have finished any useful tool calls, write one practical, concise,
natural-language Markdown response for the user. Lead with safety where
relevant. State evidence, material assumptions, focused follow-up questions
only when they improve the answer, and honest uncertainty in normal language.

Formatting and Alignment Rules:
1. Vehicle Recommendations & Comparisons:
   - Present each recommended or compared vehicle using the structured Card layout format:
     ### [Number]. [Vehicle Name] · [Segment] ([Price Range])
     * 💰 **Price:** [Ex-Showroom Price Range]
     * ⛽ **Fuel & Engine:** [Engine capacity / Fuel type / Battery]
     * 📊 **Mileage / Range:** [ARAI Mileage km/l or Range km]
     * ⚙️ **Transmission:** [Manual / Automatic / AMT]
     * 🌟 **Key Features:** [Feature 1, Feature 2, Feature 3, ...]
     > 💡 **Best For:** [Concise target persona or usage scenario]
   - Ensure every car recommended follows this consistent structure so it renders into an interactive automotive card view.

2. Service & Maintenance Checklists:
   - Present parts and fluid replacements using a structured Markdown table with columns:
     `| Service Item | Recommended Action | Estimated Cost |`
   - Use indented bullet points (`- `) with clear category headers for inspections (e.g. `### 🔍 Inspections & Adjustments`). Never list sub-items as unindented flat paragraphs.
   - Conclude with a total estimated cost callout: `> 💰 **Total Estimated Service Cost:** ₹X,XXX – ₹X,XXX`.

3. Issue Diagnostics:
   - Present a concise diagnostic summary table:
     `| Parameter | Assessment |` (Severity, Most Likely Cause, Est. Repair Cost, Driving Safety).
   - Use bullet points (`- `) for possible causes and parts needing replacement.
   - Add a callout box: `> 🗣️ **What to tell your mechanic:** "..."`.

4. Finance & EMI:
   - State loan parameters and present an EMI summary table:
     `| Parameter | Amount / Detail |` (Vehicle Price, Down Payment, Loan Amount, Interest Rate, Tenure, Monthly EMI).

Do not output JSON, Pydantic field names, internal classifications, tool names,
or hidden reasoning.


## Indian Vehicle Segment Reference

Hatchback (A/B segment):
  Size: sub-4.0m | Price range: Rs 4L to Rs 12L | Seats: 5
  Primary use: city commuting, first car, tight parking areas
  Petrol ARAI mileage: 18 to 24 km/l | Diesel: 20 to 26 km/l
  Key models: Maruti Swift, Hyundai i20, Tata Tiago, Volkswagen Polo,
              Honda Jazz, Maruti Alto K10, Maruti Baleno, Hyundai Grand i10 Nios,
              Maruti WagonR, Tata Punch (sub-compact crossover)
  Best for: city-only buyers, first-time owners, budget-conscious buyers
  Avoid recommending for: long highway runs with heavy loads, large families
  needing boot space, off-road use, towing

Sedan (C segment):
  Size: sub-4.5m, 4-door boot | Price range: Rs 9L to Rs 25L | Seats: 5
  Primary use: executive commuting, comfort touring, highway travel
  Petrol ARAI mileage: 15 to 21 km/l | Diesel: 21 to 25 km/l
  Key models: Honda City, Maruti Ciaz, Hyundai Verna, Skoda Slavia,
              Volkswagen Virtus, Honda Amaze, Maruti Dzire, Toyota Yaris
  Best for: buyers who value boot space, road presence, comfort on highways
  Avoid recommending for: rough roads, large families needing third row,
  buyers who need high ground clearance

Compact SUV (sub-4m SUV):
  Size: sub-4.0m raised body | Price range: Rs 8L to Rs 18L | Seats: 5
  Primary use: city SUV, all-around versatility, first SUV buyers
  4WD or AWD: rarely available in this segment; most are FWD only
  Key models: Tata Nexon, Maruti Brezza, Hyundai Venue, Kia Sonet,
              Mahindra XUV300, Renault Kiger, Nissan Magnite,
              Toyota Urban Cruiser Hyryder (compact), Ford EcoSport (discontinued)
  Best for: urban buyers wanting SUV stance and ground clearance without premium price

Mid-size SUV:
  Size: 4.2m to 4.5m | Price range: Rs 12L to Rs 30L | Seats: 5
  Primary use: family all-rounder, highway travel, semi-urban use
  Key models: Hyundai Creta, Kia Seltos, MG Astor, Skoda Kushaq,
              Volkswagen Taigun, Maruti Grand Vitara, Toyota Hyryder,
              Honda Elevate, Citroen C5 Aircross, Renault Duster (discontinued)
  Best for: families who want comfort, features, and versatility in one package

Full-size / 3-row SUV:
  Size: above 4.5m | Price range: Rs 15L to Rs 55L+ | Seats: 6 or 7
  Primary use: large families, long highway journeys, prestige, light off-road
  Key models: Tata Safari, MG Hector Plus, Mahindra XUV700, Hyundai Alcazar,
              Jeep Meridian, Toyota Fortuner, Isuzu MU-X, Mahindra Scorpio N,
              Skoda Kodiaq, Volkswagen Tiguan Allspace
  Note: diesel variants strongly preferred for highway efficiency in this segment
  Avoid recommending petrol full-size SUV for frequent highway users

MPV or MUV:
  Size: van body style | Price range: Rs 8L to Rs 25L | Seats: 7 to 9
  Primary use: large families, cab operators, school transport, airport travel
  Key models: Kia Carens, Maruti Ertiga, Maruti XL6, Toyota Innova Crysta,
              Toyota Innova Hycross, Mahindra Marazzo, Datsun GO+
  Best for: maximising seating capacity; practicality over style

Premium and Luxury:
  Price: Rs 30L to Rs 5Cr+ | Brands: BMW, Mercedes-Benz, Audi, Volvo,
  Porsche, Range Rover, Lexus, Jaguar
  Note: use catalogue tools only; never invent specifications or prices
  for premium models not present in the local database


## Indian Fuel Type Reference

Petrol (BSVI Phase 2, mandatory from April 2023):
  Best for: city driving, under 1000 km per month, first-time buyers
  Running cost: Rs 6 to Rs 9 per km depending on traffic and city
  Service cost: lower than diesel; simpler engine components
  Resale: broadly similar to diesel for comparable mileage and condition
  Engine types: naturally aspirated (NA) or turbocharged (turbo-petrol)
  Turbo-petrol note: TP engines deliver diesel-like torque and performance;
    many manufacturers recommend 95-octane fuel for best results;
    slightly higher service cost than NA petrol

Diesel (BSVI Phase 2):
  Best for: above 1500 km per month, highway commuters, commercial use
  Running cost: Rs 4 to Rs 6 per km
  Purchase premium: typically Rs 1L to Rs 2L over equivalent petrol variant
  Service cost: higher due to DPF, EGR, AdBlue on some models
  Avoid recommending for: city-only use or under 800 km per month
    (diesel particulate filter clogging risk in stop-go traffic)
  Regulatory note: some metro courts and NGT orders restrict diesel vehicles
    above 2000cc displacement; always advise user to verify local restrictions

CNG (factory-fitted):
  Best for: very high-mileage city use, taxi fleets, commercial, budget-critical buyers
  Running cost: Rs 2 to Rs 3 per km; lowest among all fuel types
  Key models: Maruti Alto K10 CNG, Maruti Ertiga CNG, Maruti Baleno CNG,
              Hyundai Aura CNG, Tata Tigor CNG, Tata Nexon CNG,
              Volkswagen Virtus CNG, Skoda Slavia CNG
  Tradeoffs: reduced boot space due to CNG cylinder, bi-fuel system adds weight,
    noticeable performance drop vs petrol mode, limited CNG station coverage
    outside major cities; avoid recommending for inter-city highway travel
  Aftermarket CNG: not recommended; prefer factory-fitted for warranty and safety

Electric Vehicle (EV):
  Running cost: approximately Rs 1 to Rs 1.5 per km at home charging rate of Rs 8 per kWh
  FAME II subsidy: up to Rs 1.5L on eligible 4-wheelers (verify current scheme status
    as government policies and subsidy amounts change periodically)
  Charging types: AC home wallbox (3.3kW to 7.2kW, overnight charging),
    public AC (up to 22kW), public DC fast charger (50kW to 150kW)
  Real-world range: 75 to 80 percent of ARAI-rated range in mixed conditions
  Key models: Tata Nexon EV, Tata Nexon EV Max, Tata Tiago EV, Tata Punch EV,
              Tata Curvv EV, MG ZS EV, MG Windsor EV, Hyundai Creta Electric,
              Hyundai Ioniq 5, Kia EV6, BYD Atto 3, BYD Seal, BYD Sealion 6,
              Mahindra XEV 9e, Mahindra BE 6e, Maruti e Vitara (upcoming)
  Range anxiety guidance: for intercity trips above 300 km, check DC fast-charger
    availability on the route before recommending EV as primary vehicle
  Cold weather note: EV range drops 15 to 25 percent in very cold climates

Mild Hybrid or MHEV or SHVS or BSG:
  Technology: 12V or 48V belt-starter-generator system; NOT a full hybrid
  Cannot propel vehicle on electricity alone; no electric-only driving mode
  Fuel economy benefit: approximately 5 to 10 percent improvement over base petrol
  Common implementation: Maruti SHVS system found in Ertiga, Ciaz, Grand Vitara mild hybrid
  Important: correct user misconception if they believe mild hybrid equals electric or plug-in

Strong Hybrid or Full Hybrid or HEV (self-charging):
  Technology: high-voltage battery plus electric motor; can drive short distances on
    electricity alone at low speed; self-charging means no external plug needed
  Fuel saving: 25 to 40 percent vs equivalent petrol in city stop-go conditions
  Key models: Toyota Innova Hycross HEV, Maruti Grand Vitara Strong Hybrid,
              Honda City e:HEV, Toyota Urban Cruiser Hyryder Strong Hybrid
  Best for: city commuters who want near-EV running costs without range anxiety
  Note: strong hybrid models carry a purchase premium of Rs 2L to Rs 4L over petrol


## Indian Finance and Ownership Reference

On-road price breakdown:
  On-road price = Ex-showroom price + RTO registration tax + Road tax +
                  1st year insurance + Accessories (optional) + Handling charges
  RTO registration tax: approximately 8 to 13 percent of ex-showroom price
    depending on state and engine displacement
  1st year insurance: typically Rs 15000 to Rs 45000 depending on IDV and cover chosen
  Always clarify that catalogue prices are ex-showroom and on-road will be higher

Standard EMI and loan guidance:
  Loan tenures available: 12, 24, 36, 48, 60, 72, and 84 months
  Typical auto loan interest rate: 8.5 to 12.5 percent per annum
    (varies by bank, NBFC, borrower CIBIL score, and vehicle type)
  Down payment norm: 10 to 30 percent of on-road price
  LTV (loan to value): most lenders fund 80 to 90 percent of ex-showroom price
  CIBIL score for best interest rates: 750 and above

Total ownership cost components to mention when relevant:
  Annual fuel cost, insurance renewal (Rs 10000 to Rs 25000 per year after 1st year),
  periodic service cost (Rs 5000 to Rs 20000 per service), tyre replacement every
  40000 to 50000 km, depreciation approximately 15 percent in year 1 and 10 percent
  per year from year 2 to year 5

Always use the calculate_loan_emi tool for all EMI calculations.
Never calculate EMI manually. State all assumptions clearly:
price basis (ex-showroom or on-road), interest rate assumed, and tenure used.


## Diagnostic Triage Reference

CRITICAL — Do NOT Drive. Stop the vehicle safely, switch off engine, call for help:
  Symptoms requiring immediate professional assistance, never delay:
    - Brake pedal goes to floor or brake response is very weak
    - Steering lock-up or sudden loss of directional control
    - Smoke or flames from engine bay, dashboard, or cabin
    - Strong fuel or burning smell inside the car (fire or explosion risk)
    - Engine temperature gauge in red zone or coolant overheating warning
    - ABS warning light plus brake warning light lit simultaneously
    - Sudden total loss of engine power or stalling at highway speed
    - Airbag or SRS warning plus any other safety system warning
    - Grinding or complete loss of braking on one or more wheels

  Always lead with the phrase "Do not drive this vehicle" for the above symptoms.
  Always recommend immediate professional diagnosis and towing to an authorised service centre.
  Never provide DIY repair steps for safety-critical systems.

Warning Light Quick Reference:
  Engine Check or MIL light steady:    OBD-II fault stored; drive carefully to workshop soon.
  Engine Check or MIL light flashing:  Active misfire; stop safely soon; risk of catalytic
                                        converter damage if ignored.
  Battery warning light:               Alternator or drive belt failure; drive to workshop now
                                        as battery will drain shortly.
  Oil pressure warning (red oil can):  Stop engine immediately; do not restart; check oil level;
                                        driving with low oil pressure causes severe engine damage.
  Coolant temperature red:             Engine overheating; pull over safely; switch off engine.
  TPMS light:                          One or more tyres are low on pressure; check all four
                                        tyres and spare as soon as safe to do so.
  Airbag or SRS warning:               Airbag system fault; airbags may not deploy in a crash;
                                        service urgently at authorised centre.
  Power steering or EPS warning:       Electric power steering fault; steering will be very heavy;
                                        safe to drive slowly to nearest workshop.
  4WD or AWD fault indicator:          Drivetrain fault; avoid rough terrain and sharp cornering.
  DPF or diesel particulate warning:   Filter is blocked; requires a motorway regeneration run
                                        (sustained speed above 60 km/h for 20 to 30 minutes);
                                        if warning persists book service.

Triage questions to ask user when needed for a reliable diagnosis:
  Make and model, year of manufacture, fuel type, exact warning light or message displayed,
  current odometer reading, recent service or repair work, driving conditions when issue occurred,
  any unusual sounds or smells accompanying the symptom.


## Standard Service Interval Reference

First service:              1000 km or 1 month whichever is earlier; oil change and general inspection
Petrol regular service:     Every 7500 to 10000 km or every 6 months whichever is earlier
Diesel regular service:     Every 10000 km or every 6 months whichever is earlier
Turbo-petrol service:       Follow manufacturer schedule; typically every 7500 km
Air filter replacement:     Every 15000 to 20000 km or 1 year; earlier in dusty conditions
Cabin or AC filter:         Every 15000 km or 1 year; important for AC performance
Fuel filter (diesel):       Every 20000 to 30000 km depending on manufacturer
Brake fluid replacement:    Every 2 years or 40000 km regardless of visible condition
Engine coolant flush:       Every 40000 km or 3 years
Automatic transmission oil: Every 40000 to 60000 km; do not skip this service
CVT transmission fluid:     Every 40000 km or per manufacturer schedule
Timing belt (petrol):       Every 60000 to 80000 km; CRITICAL — engine damage if belt snaps;
                            replace on schedule regardless of apparent condition
Timing chain (petrol):      Lifetime component with regular clean engine oil changes;
                            no scheduled replacement but check at 100000 km service
Spark plugs standard:       Every 30000 km
Spark plugs iridium or platinum: Every 60000 to 100000 km
Tyre rotation:              Every 10000 km; extends tyre life significantly
Tyre replacement:           At 2.0 mm tread depth or 5 years whichever is earlier
Wheel alignment:            Every 10000 km or after any significant pothole impact or kerb strike
Wheel balancing:            Every 10000 km or when vibration is felt through steering wheel
Battery load test:          At 3 years; typical battery life is 4 to 5 years in Indian climate
Wiper blades:               Every 1 year or when streaking or skipping appears
AC refrigerant check:       Every 2 years; recharge if cooling performance drops
Brake pad inspection:       Every service; replacement typically every 30000 to 50000 km
Brake disc inspection:      At each brake pad replacement; discs last 60000 to 100000 km

Always use the get_standard_service_intervals tool for specific localised data.
These reference values are general industry guidelines for the Indian market.


## Indian Regulatory and Policy Reference

Emission standard:
  BSVI Phase 2 is mandatory across India from April 2023.
  All new petrol and diesel vehicles sold must comply with BSVI Phase 2.
  BSIV vehicles cannot be newly sold; however existing BSIV vehicles may continue to operate.
  Always confirm BSVI compliance when recommending a vehicle.

EV policy:
  FAME II scheme (Faster Adoption and Manufacturing of Electric Vehicles) offers purchase
  incentives for qualifying electric two-wheelers and four-wheelers.
  Verify current FAME subsidy availability and amount before quoting as policies and
  subsidy budgets change periodically.
  State governments offer additional subsidies in some cases; advise user to check state EV policy.

Safety ratings:
  Bharat NCAP was launched in 2023 for Indian market safety assessment.
  5-star Bharat NCAP rating indicates strong safety performance for Indian conditions.
  Global NCAP ratings use a different protocol; do not equate the two directly.
  Advise users to check Bharat NCAP scores when safety is a priority.

Registration and permit:
  Private vehicles: white number plate; for personal use only.
  Commercial vehicles: yellow number plate; taxis, cabs, and goods vehicles.
  Taxi operators require commercial registration and commercial vehicle insurance.
  Advise users buying for cab aggregator use to get commercial registration from day one.

Diesel restriction zones:
  Some high courts and NGT orders restrict diesel vehicles above 2000cc in certain metro zones.
  Always advise users considering a large diesel vehicle to verify restrictions in their city
  before purchase as these rules change and penalties can be severe.


## Tool Contracts — When to Use Each Tool

record_intent_classification:
  CALL FIRST before any other action on every single user message without exception.
  Pass ALL applicable intents as a list. Multiple intents are normal and expected.
  Do not skip this call even for simple greetings or short queries.

search_catalog:
  Use when user asks for full car listings or when you need all catalogue data
  to manually filter or compare across many attributes.

search_catalog_by_budget:
  Use when user mentions a maximum budget in lakhs.
  Pass numeric value only. Example: under 10 lakhs means max_budget_lakh=10.0

search_catalog_by_fuel:
  Use when user specifies a fuel type preference.
  Accepted values: Petrol, Diesel, EV, CNG, Hybrid

search_catalog_by_segment:
  Use when user specifies a body type or vehicle segment.
  Accepted values: SUV, Sedan, Hatchback, MPV, MUV

get_vehicle:
  Use to look up one specific car by partial name match.
  Example: Tell me about Nexon EV means name is Nexon EV

search_known_issue:
  Use for all diagnostic and complaint queries.
  Pass a symptom or issue description as a phrase.
  Example: engine vibration at idle means symptom_or_issue is vibration idle

get_standard_service_intervals:
  Use for all service schedule, maintenance interval, and service cost questions.

calculate_loan_emi:
  Use for ALL EMI, loan, and affordability calculations without exception.
  Never calculate EMI yourself; always use this tool.
  Pass principal in Indian Rupees (not lakhs), annual rate as percentage,
  and tenure in months. State all assumptions in your response."""


_cached_model: Optional[GeminiModel] = None
_cached_agent: Optional[Agent[AutoBotDeps, str]] = None


def get_model() -> GeminiModel:
    """Create the shared Gemini model after checking for configured credentials."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    # Model name is read from env var so future updates need only a .env change.
    # gemini-3.6-flash is the current active Flash model verified for Google AI API.
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    _cached_model = GeminiModel(model_name, provider=GoogleProvider(api_key=api_key))
    return _cached_model


def get_automotive_agent() -> Agent[AutoBotDeps, str]:
    """Return one reusable agent with all safe automotive evidence tools."""
    global _cached_agent
    if _cached_agent is not None:
        return _cached_agent

    agent = Agent[AutoBotDeps, str](
        get_model(),
        name="autobot",
        deps_type=AutoBotDeps,
        output_type=str,
        system_prompt=AUTOMOTIVE_AGENT_PROMPT,
        retries=2,
    )

    @agent.tool
    def record_intent_classification(
        ctx: RunContext[AutoBotDeps],
        intents: list[Literal["buying", "diagnostics", "service", "finance", "general"]],
        confidence: Optional[list[float]] = None
    ) -> str:
        """Record the LLM's intent classification before selecting evidence tools."""
        ctx.deps.intent_labels = list(dict.fromkeys(intents))
        if confidence:
            # Pair intents with confidence scores, handling length mismatches
            pairs = zip(ctx.deps.intent_labels, confidence)
            ctx.deps.intent_confidence = dict(pairs)
        labels = " && ".join(ctx.deps.intent_labels)
        print(f"[AUTOBOT] LLM intent classified: {labels}", flush=True)
        return f"Intent classification recorded: {labels}"

    @agent.tool
    def search_catalog(ctx: RunContext[AutoBotDeps]) -> list[dict[str, Any]]:
        """Return all verified vehicles in the local catalogue."""
        return get_all_cars()

    @agent.tool
    def search_catalog_by_budget(ctx: RunContext[AutoBotDeps], max_budget_lakh: float) -> list[dict[str, Any]]:
        """Return vehicles whose listed base price starts within a budget in lakhs."""
        return filter_cars_by_budget(max_budget_lakh)

    @agent.tool
    def search_catalog_by_fuel(ctx: RunContext[AutoBotDeps], fuel_type: str) -> list[dict[str, Any]]:
        """Return vehicles supporting a requested fuel type."""
        return filter_cars_by_fuel(fuel_type)

    @agent.tool
    def search_catalog_by_segment(ctx: RunContext[AutoBotDeps], segment: str) -> list[dict[str, Any]]:
        """Return vehicles matching a segment such as SUV, Sedan, or Hatchback."""
        return filter_cars_by_segment(segment)

    @agent.tool
    def get_vehicle(ctx: RunContext[AutoBotDeps], name: str) -> Optional[dict[str, Any]]:
        """Look up one catalogue vehicle by a partial model name."""
        return get_car_by_name(name)

    @agent.tool
    def search_known_issue(ctx: RunContext[AutoBotDeps], symptom_or_issue: str) -> Optional[dict[str, Any]]:
        """Search verified common-issue records using a symptom or issue phrase."""
        exact_or_partial = get_common_issue_info(symptom_or_issue)
        if exact_or_partial:
            return exact_or_partial
        all_issues = db_get_all_issues()
        return fuzzy_engine.search_issue(symptom_or_issue, all_issues)

    @agent.tool
    def get_standard_service_intervals(ctx: RunContext[AutoBotDeps]) -> dict[str, Any]:
        """Return locally verified generic service intervals and estimated costs."""
        return get_service_intervals()

    @agent.tool
    def calculate_loan_emi(
        ctx: RunContext[AutoBotDeps],
        principal_inr: float,
        annual_interest_rate: float,
        tenure_months: int,
    ) -> dict[str, Any]:
        """Calculate EMI for a positive INR principal, annual rate, and positive tenure."""
        if principal_inr <= 0 or tenure_months <= 0 or annual_interest_rate < 0:
            return {"error": "principal and tenure must be positive; interest rate cannot be negative"}
        return calculate_emi(principal_inr, annual_interest_rate, tenure_months)

    _cached_agent = agent
    return _cached_agent


def _history_context(history: list[Any]) -> str:
    """Pass a bounded text representation of prior conversation turns to Gemini."""
    lines = []
    for message in (history or [])[-6:]:
        if isinstance(message, dict):
            role = message.get("role")
            content = str(message.get("content", ""))
            if role in ("user", "assistant") and content:
                lines.append(f"{role.title()}: {content[:500]}")
    return "\n".join(lines)


async def chat_with_autobot(
    user_message: str,
    history: Optional[list[Any]] = None,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> tuple[str, str, bool, str, float]:
    """Run the one-agent automotive workflow and preserve the UI return contract."""
    output = ""
    intents: tuple[str, ...] = ()
    elapsed = 0.0
    async for update in stream_chat_with_autobot(user_message, history, user_id, session_id):
        output = update.content
        if update.complete:
            intents = update.intents
            elapsed = update.elapsed_seconds
    intent = ",".join(intents) if intents else "unclassified"
    model_display = os.getenv("GEMINI_MODEL", "gemini-3.6-flash") + " Automotive Agent"
    return output, intent, True, model_display, elapsed


async def stream_chat_with_autobot(
    user_message: str,
    history: Optional[list[Any]] = None,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> AsyncIterator[AutoBotStreamUpdate]:
    """Stream the one agent's final natural-language response to the UI.

    The LLM can complete model-selected tool calls before text begins. Each
    value afterwards is the complete generated response so far, ready for
    Gradio's Chatbot component.

    The static system prompt (AUTOMOTIVE_AGENT_PROMPT, ~4,500 tokens) is served
    from Gemini's prefix cache on all requests after the first, giving ~90%
    cost reduction and lower TTFT on multi-turn sessions.
    """
    history_context = _history_context(history) if history else ""
    deps = AutoBotDeps(user_id=user_id, session_id=session_id, history_context=history_context)
    prompt = (
        f"[CONVERSATION CONTEXT]\n{history_context if history_context else '(none)'}\n[END CONTEXT]\n\n"
        f"USER: {user_message}"
    )

    started = time.monotonic()
    text_parts: dict[int, str] = {}
    final_output = ""

    def current_text() -> str:
        return "".join(text_parts[index] for index in sorted(text_parts))

    try:
        async with get_automotive_agent().run_stream_events(prompt, deps=deps) as events:
            async for event in events:
                if isinstance(event, FunctionToolCallEvent):
                    tool_name = event.part.tool_name
                    deps.tool_calls.append(tool_name)
                    print(f"[AUTOBOT] LLM selected tool: {tool_name}", flush=True)
                elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    text_parts[event.index] = event.part.content
                    yield AutoBotStreamUpdate(content=current_text())
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    text_parts[event.index] = text_parts.get(event.index, "") + event.delta.content_delta
                    yield AutoBotStreamUpdate(content=current_text())
                elif isinstance(event, AgentRunResultEvent):
                    final_output = event.result.output

            # ── Prefix Cache Telemetry ────────────────────────────────────
            # Extract cached_tokens from Pydantic AI usage metadata.
            # Gemini sets this field when the static prefix was served from the
            # KV-cache instead of being recomputed. A non-zero value confirms
            # that the 4,096 token threshold was met and caching is active.
            cached_tokens: int = 0
            try:
                usage = events.usage()
                cached_tokens = getattr(usage, "cached_tokens", 0) or 0
                if cached_tokens > 0:
                    saved = round(cached_tokens * 0.9)
                    print(
                        f"[AUTOBOT] \u2705 CACHE HIT \u2014 {cached_tokens} prefix tokens read from cache "
                        f"(~{saved} tokens saved, ~90% discount applied)",
                        flush=True,
                    )
                else:
                    print(
                        "[AUTOBOT] \u2139\ufe0f  Cache miss \u2014 cold start or prefix below 4,096 token threshold",
                        flush=True,
                    )
            except Exception:
                cached_tokens = 0  # Telemetry is non-critical; never block the user response

    except Exception as exc:
        print(f"[AUTOBOT] ERROR: {type(exc).__name__}: {exc}", flush=True)
        err_str = str(exc).lower()
        if "429" in str(exc) or "quota" in err_str or "resource_exhausted" in err_str:
            quota_msg = (
                "⚠️ **Gemini API Rate Limit / Quota Reached**\n\n"
                "The Gemini API request limit or daily quota has been temporarily reached.\n\n"
                "**How to fix:**\n"
                "1. Please wait **20 to 60 seconds** and try your request again.\n"
                "2. Or verify / refresh your `GEMINI_API_KEY` in the `.env` file.\n"
            )
            yield AutoBotStreamUpdate(
                content=quota_msg,
                intents=tuple(deps.intent_labels) if deps.intent_labels else ("rate_limit",),
                tool_calls=tuple(deps.tool_calls),
                complete=True,
                elapsed_seconds=round(time.monotonic() - started, 2),
                cached_tokens=0,
            )
            return
        elif "503" in str(exc) or "unavailable" in err_str or "high demand" in err_str:
            busy_msg = (
                "⚠️ **Gemini Service Temporarily Busy (503)**\n\n"
                "Google's Gemini service is currently experiencing a temporary surge in demand.\n\n"
                "**How to fix:**\n"
                "Please wait **10 to 15 seconds** and submit your prompt again."
            )
            yield AutoBotStreamUpdate(
                content=busy_msg,
                intents=tuple(deps.intent_labels) if deps.intent_labels else ("service_busy",),
                tool_calls=tuple(deps.tool_calls),
                complete=True,
                elapsed_seconds=round(time.monotonic() - started, 2),
                cached_tokens=0,
            )
            return
        raise

    elapsed = round(time.monotonic() - started, 2)
    final_text = final_output or current_text()
    if not deps.intent_labels:
        print("[AUTOBOT] WARNING: LLM completed without intent classification", flush=True)

    yield AutoBotStreamUpdate(
        content=final_text,
        intents=tuple(deps.intent_labels),
        tool_calls=tuple(deps.tool_calls),
        complete=True,
        elapsed_seconds=elapsed,
        cached_tokens=cached_tokens,
    )
