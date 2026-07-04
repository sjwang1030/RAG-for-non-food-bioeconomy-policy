NONFOOD_DEFINITION = """
The non-food bioeconomy refers to economic sectors, activities, and governance
measures concerning biomass, bio-based biological resources, or biomass-derived
material streams that are used, recovered, managed, processed, or mobilized for
non-food purposes.

This includes, for example:
- bioenergy, biofuels, biogas, biomethane, biomass-based heat, power, or transport fuels
- bio-based materials, biomaterials, bioplastics, wood- or fibre-based industrial materials,
  bio-based construction materials, bio-based packaging materials, pulp and paper for industrial use
- biochemicals and other bio-based chemicals
- industrial biotechnology, biomanufacturing
- biorefineries, integrated biomass conversion, and cascading biomass use
- biomass residues, organic waste, agricultural residues, forestry residues, side-streams,
  or biodegradable biomass-derived material streams that are explicitly governed for subsequent
  bioenergy, biomaterial, biochemical, or other non-food bio-based industrial use
- policies that explicitly govern biomass feedstock collection, segregation, recovery,
  preparation, mobilization, or valorization for later non-food use
- broader policy frameworks that explicitly promote, regulate, or support the use of biomass,
  bio-based products, renewable biological resources, or biomass-derived feedstocks for
  non-food energy, materials, chemicals, manufacturing, or industrial transformation

A policy should be considered relevant when it contains explicit provisions, or
contextually substantive references, linking biomass, bio-based biological resources,
bio-based products, renewable biological resources, or biomass-derived material streams
to non-food purposes.

Important exclusions:
Policies should NOT be classified as non-food bioeconomy policies if they only address:
- general waste management
- hazardous waste regulation
- general recycling
- plastic waste or plastic bag restrictions
- broad circular economy strategies
- sustainable products or ecodesign frameworks
- general climate, biodiversity, water, or development strategies
- general energy, electricity, fuel, or hydrogen policy

unless the text explicitly and substantively links biomass, bio-based resources,
bio-based products, biodegradable biomass-derived material streams, or biomass-derived
feedstocks to non-food energy, materials, chemicals, industrial processing,
manufacturing, or other non-food uses.

Policies exclusively addressing:
- food production
- agriculture productivity
- nutrition
- food security
- feed

without explicit and substantive non-food biomass provisions should be excluded.

Methodological boundary principle:
This screening follows the boundary-setting logic used in global bioeconomy policy
reviews: bioeconomy-related policies must have a strong link to bioeconomy development,
especially in biotechnology, bioenergy, biomass, or the biobased economy/industry.

For traditional bioeconomic domains such as agriculture, forestry, marine resources,
primary production, and for broader strategies on research and innovation,
sustainability, green growth, blue growth, circular economy, climate, waste, or
general energy transition, relevance should NOT be assumed from the policy domain
alone. These policies qualify only when they explicitly prioritize bioeconomy
development for non-food use, non-food biomass use, or innovative biobased
approaches for non-food use through a concrete policy measure.

Because this study focuses on the non-food bioeconomy, the prioritized bioeconomy
content must be linked to non-food energy, fuels, heat, power, materials, chemicals,
industrial biotechnology, biorefineries, manufacturing, construction, biomass
residue valorization, or feedstock recovery for later non-food use.

Important clarification:
A policy does not need to be exclusively labeled as a "bioeconomy policy" to be relevant.
However, broad climate, industrial, forestry, circular economy, or innovation documents
count only when they contain at least one concrete policy measure explicitly directed at
non-food bio-based deployment or governance.
""".strip()


NON_FOOD_PROMPT_VERSION = "2026-05-08-v14-gbpr-boundary"


NON_FOOD_SYSTEM_PROMPT = (
    "You are a strict policy classifier for non-food bioeconomy screening. "
    "Use only the provided text. "
    "Treat the primary policy object as a strong but overridable signal. "
    "A document qualifies when it either (a) mainly governs a non-food bio-based value chain, "
    "or (b) contains a substantive dedicated section or block of text whose operative focus "
    "is a non-food bio-based object with at least one concrete current or forward-looking "
    "governance measure. "
    "Do not count subordinate mentions, listed items, restricted activities, examples, "
    "definitions, or background discussion inside another policy domain. "
    "Do not treat retrospective progress reports or past-achievement summaries as governance "
    "unless the text also contains a current or forward-looking measure, target, program, "
    "requirement, or dedicated clause for the non-food bio-based object. "
    "When evidence is weak but readable, choose no_non_food. "
    "Use unclear only for genuinely insufficient text. "
    "Return strict JSON only."
)


def build_non_food_prompt(text: str) -> str:
    return f"""
You are an expert in bioeconomy policy analysis.

Definition of non-food bioeconomy:

{NONFOOD_DEFINITION}

Task:
Classify whether the policy contains direct and substantive non-food bioeconomy policy content.

Core rule:
Classify as contains_non_food only when the text shows a direct and substantive
policy link between a non-food bio-based object and a real governance measure.

The policy must either:
(a) directly govern a non-food bio-based value chain, feedstock, product, process,
or end use; or
(b) be a broader policy that explicitly prioritizes non-food biomass use,
non-food bioeconomy development, or innovative biobased approaches for non-food
use through a substantive and operative policy component.

A broad policy domain is not enough. Policies on agriculture, forestry, marine
resources, climate, circular economy, green growth, blue growth, innovation,
waste, renewable energy, or general development should be classified as
contains_non_food only when the non-food bio-based content is explicitly
prioritized and attached to a concrete governance measure.

Mere references to biomass, renewable energy, residues, forestry, agriculture,
circularity, biotechnology, or sustainability are insufficient.

Apply these four checks in order:

1. Primary policy object (strong but overridable signal)
Ask: does the document mainly govern a non-food bio-based value chain, feedstock, product,
process, or end use?
- If YES -> proceed directly to Check 3.
- If NO -> the document belongs to another primary domain (biodiversity, climate, food, planning,
  energy efficiency, etc.). Do NOT automatically exclude. Instead apply Check 2.

2. Substantive non-food component in broader policies
A broader document qualifies when it contains a substantive block of text - a dedicated numbered
article, a named section, or a coherent multi-sentence passage - whose own operative focus
explicitly prioritizes non-food biomass use, non-food bioeconomy development, or an innovative
biobased approach for non-food use, AND that block contains at least one concrete current or
forward-looking governance measure.

A "substantive block" means the non-food bio-based content has its own operative logic: it sets
a rule, target, deadline, program, guidance task, review mandate, funding mechanism, standard,
authorization rule, or deployment action specifically for the non-food bio-based object. A section
heading alone is not enough; there must be operative content beneath it.

This override does NOT apply when the bio-based content is only:
- one item in a list of fuel types, energy sources, or technology categories
- a restricted or prohibited activity requiring consent
- a definitional or technical category used to describe scope
- a passing mention or single sentence without a governance measure attached
- a reference to another document's rules without adding new governance

Important: multiple incidental mentions do not add up to a substantive section.
If bio-based terms appear in several different lists or clauses but each appearance is
itself a listed item, a permitted land use, a facility type, or a technical category,
the accumulation of those appearances does not satisfy Check 2.
Each mention must be evaluated on its own role in the text. Frequency of bio-based
keywords is not a substitute for a dedicated operative block.

Special exclusion rule for broad and traditional sectors:
For agriculture, forestry, marine, livestock, manure, fertilizer, food/feed,
circular economy, plastic waste, waste management, climate, green growth,
blue growth, sustainability, innovation, renewable energy, electricity, gas,
or general development policies, choose no_non_food unless the text explicitly
prioritizes a non-food bio-based pathway and provides a concrete governance
measure for that pathway.

Do not infer relevance from sector proximity. For example:
- agriculture is not enough unless it governs biomass residues, bioenergy,
  biobased materials, biorefineries, or other non-food uses
- forestry is not enough unless it governs forest biomass, wood-based materials,
  mass timber, forest bioeconomy, pulp/cellulose/lignin valorization, or
  bioenergy for non-food use
- circular economy is not enough unless it governs biobased products,
  biodegradable biomass-derived materials, bio-waste digestion, composting,
  biomass residue valorization, or feedstock recovery for non-food use
- renewable energy is not enough unless it explicitly governs bioenergy,
  biomass, biofuels, biogas, biomethane, or bioliquids

3. Concrete non-food governance content
The text must contain at least one concrete current or forward-looking governance measure for
the non-food bio-based object, such as:
- a rule, requirement, standard, or obligation directed at a specific actor or activity
- a quantitative or qualitative target with a timeframe
- an authorization, permit regime, or certification scheme
- a funding mechanism, tax incentive, or subsidy program
- an implementation program, deployment task, or R&D agenda with named deliverables
- a collection rule, recovery rule, or mobilization rule
- a review mandate, guidance development task, or policy revision plan with a named deadline

The following do NOT satisfy Check 3 - even if bio-based terms appear prominently:
- aspirational or opportunity language: "should", "can", "potential", "opportunities",
  "have the potential to", "offer opportunities for"
- sector descriptions or market characterizations without an operative rule
- general statements of intent without a named program, deadline, or requirement
- references to another document's rules without adding new obligations in this text
- retrospective statistics, capacity figures, or investment summaries from past periods
- a single sentence describing what actors could or may do with no obligation attached

A governance measure must be operative: it must impose, authorize, fund, require, or commit
to something specific. Descriptive or aspirational language alone does not qualify.

4. Non-food use linkage
The governed object must be clearly linked to non-food energy, fuels, heat, power, materials,
chemicals, industrial processing, manufacturing, construction, or later feedstock recovery for
those uses.

Calibration examples - use these to anchor your judgment:

CONTAINS (primary object path):
- A biofuel blending mandate regulation -> primary object is biofuel -> contains
- A biogas plant exemption order -> primary object is biogas -> contains
- A sustainability criteria regulation for biofuels and bioliquids -> contains

CONTAINS (substantive section override path):
- EU Biodiversity Strategy: contains a named section on bioenergy with a Commission review
  mandate, a phase-out trajectory for high-ILUC biofuels, and an operational guidance task -
  each with a named 2020/2021 deadline -> substantive section with concrete measures -> contains
- Waste Directive transposition: contains a dedicated numbered article "Bio-waste" that sets
  a collection, composting, and digestion obligation -> dedicated clause with operative content -> contains
- Food/agri strategy with dedicated bioenergy chapter: contains a named chapter with
  forward-looking investment targets, scheme references, or program commitments for bioenergy
  or biomass crops -> substantive section with operative content -> contains
- Agriculture or rural development strategy: contains a dedicated bioenergy or biomass-residue
  section with investment targets, biogas deployment programs, residue collection rules, or
  subsidies for biomass-to-energy/materials use -> contains_non_food with basis
  substantive_broad_framework
- Circular economy strategy: contains a dedicated measure to promote biobased products,
  bioplastics, biomass-derived packaging, bio-waste digestion, or biomass residue valorization
  for industrial non-food use -> contains_non_food with basis substantive_broad_framework
- Forest strategy: contains concrete measures for forest bioeconomy, wood-based industrial
  materials, mass timber construction, lignin/cellulose valorization, or forest biomass for
  non-food energy/materials -> contains_non_food with basis substantive_broad_framework

NO_NON_FOOD (incidental mention):
- Wild Birds Special Protection Area regulation: lists "planting of multi-annual bioenergy crops"
  as a restricted activity requiring ministerial consent -> bio-based content is a restricted item,
  not a governed value chain -> no_non_food
- Building energy performance methodology: mentions "biomass heating boiler" as one of many
  technical input parameters in an energy calculation formula -> listed technical category,
  no governance measure for biomass -> no_non_food
- Climate white paper reporting past achievements: reports past biomass energy capacity figures,
  historical investment summaries, and past utilization totals with no forward-looking target,
  program, or obligation -> retrospective statistics only, Check 3 not satisfied -> no_non_food
- Farm to Fork Strategy: contains sentences such as "farmers should grasp opportunities to
  develop biogas production" and "farms have the potential to produce biogas" - these use
  aspirational and opportunity language ("should", "potential", "opportunities") with no
  operative rule, named program, deadline, or funding commitment directed at biogas or biomass ->
  aspirational description only, Check 3 not satisfied -> no_non_food
- Territory planning regulation where bio-based terms appear only in land-use permission
  lists: "biomass growing" listed alongside plant production and vegetable growing as
  permitted agricultural activities in road/railway planning zones; "biogas cogeneration
  units" listed alongside waste landfill sites and mineral extraction sites as large
  engineering structures subject to general spatial planning requirements -> each appearance
  is a listed item in a different clause; multiple listed-item appearances do not combine
  into a substantive section; Check 2 and Check 3 both not satisfied -> no_non_food
- Agriculture strategy: promotes agricultural productivity, food security, animal feed, or
  farm income, and only mentions residues or biogas as possible opportunities without a named
  program, target, subsidy, standard, or obligation for non-food biomass use -> no_non_food
- Forestry programme: focuses on forest conservation, biodiversity, forest management, or rural
  development, and does not prioritize wood-based materials, forest bioeconomy, biomass energy,
  mass timber, or forest biomass valorization through concrete measures -> no_non_food
- Circular economy strategy: addresses recycling, waste prevention, plastics, product design, or
  resource efficiency in general, but does not explicitly prioritize biobased products,
  bioplastics, bio-waste digestion, biomass residue valorization, or feedstock recovery for
  non-food use -> no_non_food
- Renewable energy or climate strategy: lists biomass, bioenergy, or biofuels as one energy
  source among many, but provides no target, mandate, subsidy, standard, deployment plan, or
  sustainability requirement specifically for the bio-based object -> no_non_food

Decision rules:
- contains_non_food = Checks 1 or 2 are satisfied, AND Checks 3 and 4 are satisfied
- no_non_food = the text is readable but the policy link is absent, subordinate, incidental,
  restricted, or only retrospective
- unclear = the text is too incomplete or ambiguous to resolve

Basis rules:
- explicit_nonfood_target:
  use when the document directly governs a non-food bio-based object as a central policy object
- substantive_broad_framework:
  use when a broader framework or broader law contains a substantive section or block with a
  concrete qualifying non-food measure (override path)
- incidental_or_excluded_mention:
  use when the bio-based terms are only incidental, subordinate, descriptive, restricted, or
  retrospective
- no_relevant_signal:
  use when no meaningful non-food bio-based signal is present
- mixed_or_borderline:
  use only when there is some potentially relevant signal but the required policy link remains
  unproven
- insufficient_text_or_ambiguous:
  use only when the text quality is too poor or incomplete to judge reliably

Output requirements:
- matched_terms must be short exact phrases copied from the text, not paraphrases
- matched_terms should contain 0 to 3 items
- reason must be brief and grounded in the text, and must name the specific measure that
  satisfies Check 3

Return JSON:
{{
  "category": "contains_non_food | no_non_food | unclear",
  "confidence": "high | medium | low",
  "basis": "explicit_nonfood_target | substantive_broad_framework | incidental_or_excluded_mention | no_relevant_signal | mixed_or_borderline | insufficient_text_or_ambiguous",
  "matched_terms": ["exact phrase 1", "exact phrase 2", "exact phrase 3"],
  "reason": "brief explanation grounded in the text"
}}

Policy text:

{text}
""".strip()
