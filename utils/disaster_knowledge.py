"""
Disaster Safety, Causes, and Emergency Rescue Knowledge Base
============================================================
Comprehensive domain-expert repository detailing:
- Geological & Meteorological root causes of natural disasters.
- Step-by-step structural search and rescue protocols ("How to Save Someone").
- Actionable precautionary guidelines across Pre-, During-, and Post-disaster phases.
- Emergency checklists and survival kit recommendations.
"""

from typing import Dict, List, Any

DISASTER_TYPES = [
    "Earthquake",
    "Hurricane / Cyclone",
    "Wildfire",
    "Tsunami",
    "Flood",
    "Volcanic Eruption",
]

DISASTER_DETAILS: Dict[str, Dict[str, Any]] = {
    "Earthquake": {
        "icon": "🏚️",
        "scientific_name": "Seismic Ground Displacement & Fault Rupture",
        "color": "#e74c3c",
        "causes": [
            "Tectonic Plate Subduction: Gradual accumulation of immense elastic strain energy at tectonic plate boundaries (convergent, divergent, transform).",
            "Fault Rupture & Slip: When shear stress exceeds rock friction, sudden catastrophic slip releases seismic energy as P (primary) and S (secondary) shear waves.",
            "Soil Liquefaction: Saturated, loosely packed sandy or silty soil loses shear strength due to ground shaking, behaving like a viscous liquid and undermining foundation footings.",
            "Volcanic & Anthropogenic Triggers: Magma movement inside subterranean chambers or deep wastewater injection / deep mining collapses.",
        ],
        "structural_impact": "Causes severe lateral shear stress, soft-story collapses, pancake floor collapses, foundation detachment, and vertical load-bearing column failure.",
        "rescue_protocols": [
            "Scene Safety & Secondary Collapse Assessment: Never enter an unstable collapse zone without shoring or cribbing. Assess overhead hazards (hanging slabs, snapped rebar, live wires).",
            "The Golden Hour Rule: 80% of live rescues occur within the first 24-48 hours. Prioritize void spaces (under heavy lintels, bathtubs, reinforced doorframes).",
            "START Triage Protocol: Classify victims immediately into Immediate (Red), Delayed (Yellow), Walking Wounded (Green), and Deceased/Expectant (Black).",
            "Crush Syndrome Prevention: If a victim's limb has been pinned under heavy debris for > 2 hours, do NOT rapidly release the weight without administering intravenous fluids; sudden reperfusion can release lethal potassium and myoglobin into the bloodstream causing cardiac arrest or acute renal failure.",
            "Hasty Acoustic & Visual Search: Use call-and-listen patterns (whistles, tapping on structural steel/pipes) in 15-minute quiet windows across the disaster field.",
        ],
        "precautions": {
            "before": [
                "Structural Retrofit: Anchor house frames to concrete foundations with hold-downs; strap water heaters and heavy bookcases to wall studs.",
                "Survival Kit: Pack 72-hour survival kit with 4 liters of water/person/day, non-perishable rations, NOAA crank radio, N95 masks, and heavy work gloves.",
                "Emergency Drills: Identify interior safe spots away from exterior windows, unreinforced chimneys, and hanging lighting fixtures.",
            ],
            "during": [
                "Drop, Cover, and Hold On: Drop to hands and knees immediately. Cover head and neck beneath a sturdy desk or table. Hold on until shaking ceases completely.",
                "If Outdoors: Move to open ground away from electrical pylons, unreinforced masonry facades, glass curtain walls, and overpasses.",
                "Do NOT Run to Doorways or Elevators: Modern interior doorframes offer no structural protection over desks, and elevators may lose power or jam in hoistways.",
            ],
            "after": [
                "Check for Gas Leaks: Sniff for rotten egg sulfur odor. If suspected, shut off main gas valve using an emergency wrench; never flip electrical switches (sparks ignite gas).",
                "Inspect Structural Cracks: Avoid buildings with diagonal X-cracks in shear walls or bowed exterior columns.",
                "Prepare for Aftershocks: Expect strong seismic aftershocks within minutes, hours, or days following the main shock.",
            ],
        },
    },
    "Hurricane / Cyclone": {
        "icon": "🌀",
        "scientific_name": "Tropical Cyclonic Depression & Storm Surge",
        "color": "#3498db",
        "causes": [
            "Sea Surface Temperature > 26.5°C: Warm ocean waters evaporate rapidly, feeding massive thermal energy and convective updrafts into the low-pressure vortex.",
            "Coriolis Deflection: Earth's rotation bends incoming air currents into intense cyclonic circulation (counter-clockwise in Northern Hemisphere).",
            "Low Vertical Wind Shear: Minimal high-altitude wind variation allows the towering eye-wall thunderstorm columns to maintain vertical symmetry.",
            "Ocean Storm Surge: High winds push vast mounds of seawater onshore, heightened by astronomical high tides and atmospheric pressure drop.",
        ],
        "structural_impact": "Windward wall pressure causes roof uplift via aerodynamic Bernoulli effect; high-velocity windblown missile debris shatters fenestration, causing internal pressurization blowouts.",
        "rescue_protocols": [
            "Water Rescue in Swift Currents: Always approach stranded victims from upstream or tethered boats; deploy high-buoyancy PFDs (Personal Flotation Devices).",
            "Electrocution Hazard Isolation: Submerged transformers and submerged house distribution panels charge floodwaters with lethal voltage. Ensure grid power cut before wading.",
            "Rooftop / Attic Extraction: Never trap victims in an unventilated attic without an axe to breach through the roof deck if water levels surge.",
            "Hypothermia Management: Prolonged exposure to wind and water induces rapid hypothermia even in warm climates. Strip wet clothes and apply emergency foil blankets.",
        ],
        "precautions": {
            "before": [
                "Install Hurricane Straps: Tie roof trusses directly down to exterior bearing walls with galvanized steel hurricane clips.",
                "Impact-Rated Shutters: Fasten 5/8-inch marine plywood or motorized metal storm shutters over all windows and sliding patio doors.",
                "Clear Perimeter: Remove dead trees and anchor all loose garden furniture, antennas, and propane cylinders.",
            ],
            "during": [
                "Shelter in Interior Core: Retreat to an interior room, hallway, or walk-in closet on the lowest floor above anticipated flood surge without windows.",
                "Beware the Eye: The passage of the storm's calm eye is deceptive; violent winds will reverse direction abruptly within 30-60 minutes.",
                "Avoid Candle Use: High winds extinguish candles and ignite curtains. Use LED lanterns and battery floodlights.",
            ],
            "after": [
                "Avoid Standing Floodwater: Contains raw sewage, toxic chemical runoff, broken glass, submerged venomous reptiles, and downed lines.",
                "Verify Generator Ventilation: Run petrol generators only outdoors at least 20 feet away from open windows to prevent carbon monoxide poisoning.",
                "Photograph Damage for Insurance: Document structural loss from outside before clearing debris or removing flood-soaked drywall.",
            ],
        },
    },
    "Wildfire": {
        "icon": "🔥",
        "scientific_name": "Uncontrolled Wildland-Urban Interface (WUI) Firestorm",
        "color": "#e67e22",
        "causes": [
            "Prolonged Aridity & Heat: Severe drought drops fuel moisture content of timber, scrub, and brush to under 5%.",
            "Diablo & Santa Ana Katabatic Winds: Dry, down-slope desert winds gusting over 60-80 mph generate rapid flame propagation and long-range ember casting (spotting).",
            "Ignition Vectors: Downed transmission powerlines, lightning strikes, machinery sparks, or reckless open campfires.",
            "Pyrocumulonimbus Clouds: Massive firestorms create their own local meteorological weather, generating dry lightning and fire tornadoes.",
        ],
        "structural_impact": "Airborne burning embers penetrate attic soffit vents, igniting structural roof trusses from within; radiant thermal flux shatters exterior glass.",
        "rescue_protocols": [
            "Burn & Inhalation Triage: Administer 100% humidified oxygen for victims with smoke inhalation; check airway for soot, singed nasal hairs, and stridor.",
            "Thermal Burn Dressing: Cover partial and full-thickness burns with clean, dry, sterile dressings. Avoid direct ice application to prevent hypothermic tissue necrosis.",
            "Escape Route Establishment: Always maintain an 'anchor point' and two independent escape routes to an established safety zone (cleared mineral soil or water body).",
            "Structure Fire Sheltering: If caught in fire front, shelter inside a non-combustible building or fire shelter tent rather than running through open brush.",
        ],
        "precautions": {
            "before": [
                "Defensible Space Zones: Maintain Zone 0 (0-5 ft: ember-resistant buffer with zero combustible mulch/shrubs) and Zone 1 (5-30 ft: lean, clean, green landscaping).",
                "Ember-Resistant Mesh: Install 1/8-inch metal mesh over all attic, soffit, and foundation vents to block flying firebrands.",
                "Class-A Fire Roofing: Use metal, clay tile, or asphalt composite shingles; seal all roof valleys and clear pine needles weekly.",
            ],
            "during": [
                "Evacuate at 'Go' Order: Never hesitate; vehicle gridlock occurs rapidly once smoke reduces visibility to zero.",
                "Pre-Evac Prep: Shut all exterior windows and metal shutters; turn off main propane/natural gas supply; leave interior lights ON for firefighter visibility.",
                "Don Protective Apparel: Wear 100% heavy cotton or wool garments, leather work boots, goggles, and an N95/P100 respirator mask (synthetic clothes melt onto skin).",
            ],
            "after": [
                "Check Roof and Attic: Inspect eaves and attic insulation for smoldering embers for at least 24 hours after fire front passes.",
                "Beware of Ash Hazards: Wet ash is caustic (contains potassium hydroxide) and damages skin/lungs; wear gloves and eye protection.",
                "Watch for Burned Tree Hazards: 'Widow-maker' weakened trees and smoldering root cavities can collapse days after surface fires subside.",
            ],
        },
    },
    "Tsunami": {
        "icon": "🌊",
        "scientific_name": "Mega-Wavelength Megathrust Oceanic Surge",
        "color": "#1abc9c",
        "causes": [
            "Subsea Megathrust Earthquakes: Vertical seabed displacement along tectonic subduction faults displaces thousands of cubic kilometers of water column.",
            "Submarine Volcanic Caldera Collapse: Explosive volcanic flank failures or caldera implosions generate colossal hydro-acoustic waves.",
            "Undersea Landslides: Continental shelf debris slides rapidly displace overlying water volumes with minimal seismic warning.",
            "Bathymetric Shoaling: As deep-ocean high-speed waves (500+ mph) approach shallow coastline, wave speed slows while wave height stacks exponentially.",
        ],
        "structural_impact": "Hydrodynamic wave impact delivers immense horizontal kinetic pressure; buoyant uplift lifts buildings off slabs; scouring erosion undermines foundation pilings.",
        "rescue_protocols": [
            "Immediate Vertical Relocation: Move victims to minimum 4th floor or higher of reinforced concrete structures if ground evacuation to 100ft elevation is unachievable.",
            "Floating Debris Trapping: Victims trapped in tsunami waters suffer blunt force trauma from battered cars, shipping containers, and lumber; use safety lines and throw-bags.",
            "Aspiration & Secondary Drowning: Water in lungs causes delayed pulmonary edema hours after rescue. Any victim who inhaled floodwater requires immediate clinical observation.",
            "Multiple Wave Series: Never declare all-clear after the first wave. Tsunami wave trains often arrive in surges 15-60 minutes apart, with 2nd or 3rd waves frequently being the largest.",
        ],
        "precautions": {
            "before": [
                "Map Tsunami Evacuation Zones: Know primary and secondary pedestrian routes to elevated ground (> 30 meters / 100 feet above sea level).",
                "Heed Natural Warnings: A prolonged severe earthquake (> 20 seconds) or rapid unusual ocean shoreline receding is nature's immediate tsunami warning.",
                "Sign up for Coastal Siren Alerts: Keep cellular and coastal siren notifications active at all times.",
            ],
            "during": [
                "Evacuate on Foot: Never take cars during sudden tsunami evacuation; road gridlock traps motorists in the wave path.",
                "Do Not Watch from the Beach: If you can see the approaching wave crest, you are already too close to outrun its speed.",
                "Climb High: If trapped, climb an inland hill or the highest structural reinforced concrete hotel/parking garage.",
            ],
            "after": [
                "Stay Away from Coastal Inlets: Strong harbor currents and violent reflux continue for hours after the main wave crests.",
                "Boil Water Advisories: Water pipelines are ruptured and cross-contaminated with saltwater and industrial effluents.",
                "Wait for Official Civil Defense All-Clear: Do not return to low-lying areas until emergency authorities state all wave series have completed.",
            ],
        },
    },
    "Flood": {
        "icon": "🌊",
        "scientific_name": "Hydrological Inundation & Riverine Overflow",
        "color": "#2980b9",
        "causes": [
            "Atmospheric Rivers & Persistent Cloudbursts: Stalled high-moisture fronts dumping hundreds of millimeters of rain in short timeframes.",
            "Riverbank & Levee Overtopping: Watershed drainage basins exceed capacity, breaching riverbanks and artificial flood walls.",
            "Urban Impervious Surface Runoff: Paved concrete surfaces prevent natural ground infiltration, overwhelming municipal stormwater catch basins.",
            "Rapid Snowmelt & Ice Jams: Spring warm fronts thawing mountain snowpacks combined with rain-on-snow events.",
        ],
        "structural_impact": "Hydrostatic hydrostatic pressure buckles basement walls; saturated foundation subsoil loses bearing capacity; drywall and insulation wick toxic moisture causing black mold.",
        "rescue_protocols": [
            "Reach, Throw, Row, Go: Practice the water rescue hierarchy. Never swim to a victim unless trained and equipped with a rescue harness.",
            "Tethered Flotation Rescues: Use flat-bottom rescue rafts and personal protective drysuits; secure safety upstream spotters.",
            "Decontamination Protocol: Thoroughly wash victims in clean water; floodwaters harbor Leptospirosis, E. coli, and industrial toxins.",
            "Thermal Shock Mitigation: Cold water immersion causes rapid hyperventilation and loss of limb dexterity; treat for mild to severe hypothermia.",
        ],
        "precautions": {
            "before": [
                "Elevate Critical Utilities: Raise electrical breaker panels, water heaters, and HVAC compressors above the Base Flood Elevation (BFE).",
                "Install Sump Pumps & Backflow Valves: Check valves on sewer lines prevent municipal sewer backflow into your ground floor.",
                "Flood Barriers: Store sandbags, inflatable water barriers, or flood gates to protect door thresholds.",
            ],
            "during": [
                "Turn Around, Don't Drown: Just 6 inches of swift water can sweep an adult off their feet; 12 inches can float a car; 24 inches carries away SUVs.",
                "Disconnect Electrical Mains: Shut off the central circuit breaker before water reaches ground sockets (only if the panel area is dry).",
                "Move Valuables to Upper Floors: Secure essential identification, passports, prescription medicines, and electronics on the highest level.",
            ],
            "after": [
                "Ventilate and Dry Structure: Open doors and windows; run industrial dehumidifiers to dry framing lumber within 48 hours to halt mold growth.",
                "Discard Contaminated Food: Any food item that touched floodwater must be disposed of immediately, including canned goods with compromised seams.",
                "Inspect Foundation Footings: Check for ground sinkage, cracked exterior brickwork, or tilted support posts before reinhabiting.",
            ],
        },
    },
    "Volcanic Eruption": {
        "icon": "🌋",
        "scientific_name": "Magmatic Fragmentation, Pyroclastic Flow & Tephra Fall",
        "color": "#c0392b",
        "causes": [
            "Magma Chamber Overpressurization: Ascending low-density silica-rich magma builds gas pressure (water vapor, CO2, SO2) until volcanic crust shatters.",
            "Hydrovolcanic (Phreatic) Explosions: Groundwater contacts hot subterranean magma, causing instantaneous flash steam explosions.",
            "Volcanic Dome Collapses: Gravitational collapse of unstable viscous lava domes triggers devastating supersonic pyroclastic density currents (PDCs).",
            "Lahars (Volcanic Mudflows): Volcanic heat rapidly melts glacier caps or heavy rainfall washes loose volcanic tephra into boiling mud torrents.",
        ],
        "structural_impact": "Heavy accumulated volcanic ash loads (> 100 kg/m²) collapse broad-span roofs; incandescent pyroclastic flows incinerate wooden structures instantly.",
        "rescue_protocols": [
            "Respirator & Eyewear Distribution: Provide high-efficiency N95/N99 respirators and airtight goggles; volcanic ash consists of razor-sharp pulverized rock and glass.",
            "Thermal Blast First Aid: Treat victims of volcanic blast with high-degree burn care; monitor for toxic gas inhalation (sulfur dioxide, hydrogen fluoride).",
            "River Valley Evacuation: Avoid all riverbeds and canyon bottoms within 30 miles of volcano due to high-speed lahar surges.",
            "Avoid Ash-Clogged Vehicles: Vehicle combustion engines choke rapidly on volcanic ash; keep air intake filters clean or navigate on foot.",
        ],
        "precautions": {
            "before": [
                "Monitor Volcano Alert Levels: Follow geological observatory bulletins for micro-seismicity and sulfur dioxide plume emissions.",
                "Stock Goggles and Ash Masks: Ensure multiple particulate masks per family member; volcanic ash causes corneal abrasions and severe bronchospasm.",
                "Roof Clearing Tools: Procure sturdy ladders, brooms, and scrapers to sweep ash off roofs before structural load thresholds are reached.",
            ],
            "during": [
                "Seek Immediate Indoor Shelter: Close all windows, doors, chimney dampers, and turn off heating/cooling ventilation systems.",
                "Protect Respiratory Tract: If caught outdoors, cover mouth and nose with a damp cloth or certified particulate respirator mask.",
                "Stay Clear of Low-Lying Drainages: Lahars travel downstream at 40-60 mph with the consistency of wet concrete.",
            ],
            "after": [
                "Carefully Clear Roof Ash: Sweep ash off roofs in layers; do not hose down ash with water as wet ash increases roof weight by 100-200%.",
                "Filter Drinking Water: Ash contaminates municipal reservoirs with fluorine and heavy metals; drink bottled water until certified potable.",
                "Wear Long Protective Clothing: Wear full-sleeve garments and gloves during cleanup to avoid abrasive skin irritation.",
            ],
        },
    },
}

EMERGENCY_SUPPLIES_CHECKLIST = [
    {"item": "Potable Water (4 Liters per person/day for 3-5 days)", "category": "Survival Essentials", "priority": "Critical"},
    {"item": "Non-Perishable High-Calorie Rations (Energy bars, MREs, canned food)", "category": "Survival Essentials", "priority": "Critical"},
    {"item": "Manual Can Opener & Multi-Tool (Pliers, wire cutter, blade)", "category": "Tools", "priority": "High"},
    {"item": "Comprehensive First-Aid Kit (Trauma bandages, tourniquet, antiseptic, splints)", "category": "Medical", "priority": "Critical"},
    {"item": "N95 / P100 Particulate Respirator Masks & Airtight Goggles", "category": "Protection", "priority": "High"},
    {"item": "Battery / Solar / Hand-Crank NOAA Weather Alert Radio", "category": "Communication", "priority": "High"},
    {"item": "High-Output LED Flashlights & Chemical Glowsticks", "category": "Lighting", "priority": "Medium"},
    {"item": "10,000+ mAh Portable Power Bank with USB cables", "category": "Communication", "priority": "High"},
    {"item": "Emergency Mylar Thermal Blankets & Heavy Wool Ponchos", "category": "Warmth", "priority": "Medium"},
    {"item": "Waterproof Document Pouch (Passports, insurance deeds, prescriptions)", "category": "Documents", "priority": "High"},
    {"item": "Heavy-Duty Leather Work Gloves & Safety Whistle", "category": "Search & Rescue", "priority": "High"},
    {"item": "Water Purification Tablets or 0.1-Micron Hollow Fiber Filter", "category": "Survival Essentials", "priority": "High"},
]
