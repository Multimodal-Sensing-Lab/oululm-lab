import csv, random, itertools
random.seed(42)

# name, scientific, phylum, class, order, diet, habitat, thermo, resp, circ, repro, symmetry, waste, traits
A = []
def add(*a): A.append(dict(zip("name sci phylum cls order diet habitat thermo resp circ repro sym waste traits".split(), a)))

M="Chordata"; MAM="Mammalia"; AV="Aves"; REP="Reptilia"; AMP="Amphibia"
add("African bush elephant","Loxodonta africana",M,MAM,"Proboscidea","herbivore","savanna and woodland","endothermic","lungs","closed, with a four-chambered heart","viviparous, with a gestation period of about 22 months","bilateral","urea",
 ["the trunk is a muscular hydrostat formed by the fusion of the nose and upper lip","its large ears act as heat radiators through dense networks of blood vessels","its molars are replaced horizontally from the back of the jaw several times during life","it can detect low-frequency infrasonic vibrations through its feet"])
add("blue whale","Balaenoptera musculus",M,MAM,"Artiodactyla","filter-feeding carnivore that consumes mainly krill","open ocean","endothermic","lungs","closed, with a four-chambered heart","viviparous","bilateral","urea",
 ["it uses baleen plates made of keratin to strain krill from engulfed seawater","ventral throat pleats allow the buccal cavity to expand enormously during lunge feeding","a thick layer of blubber provides insulation and energy storage","its blowhole is a pair of nostrils displaced to the top of the skull"])
add("gray wolf","Canis lupus",M,MAM,"Carnivora","carnivore","forest, tundra, and grassland","endothermic","lungs","closed, with a four-chambered heart","viviparous","bilateral","urea",
 ["its carnassial teeth are specialized for shearing flesh","it lives in social packs usually organized around a breeding pair","its olfactory epithelium contains far more receptor cells than that of humans"])
add("lion","Panthera leo",M,MAM,"Carnivora","carnivore","savanna and grassland","endothermic","lungs","closed, with a four-chambered heart","viviparous","bilateral","urea",
 ["it is the only highly social species of the family Felidae","an elongated vocal fold structure allows it to produce loud roars","females in a pride often synchronize reproduction and nurse cubs communally"])
add("house mouse","Mus musculus",M,MAM,"Rodentia","omnivore","human-associated and agricultural habitats","endothermic","lungs","closed, with a four-chambered heart","viviparous, with a gestation period of about 19 to 21 days","bilateral","urea",
 ["its incisors grow continuously and are kept short by gnawing","it is one of the most widely used model organisms in genetics and biomedical research","its high surface-area-to-volume ratio imposes a high mass-specific metabolic rate"])
add("bottlenose dolphin","Tursiops truncatus",M,MAM,"Artiodactyla","carnivore","coastal and offshore marine waters","endothermic","lungs","closed, with a four-chambered heart","viviparous","bilateral","urea",
 ["it uses echolocation, emitting clicks that are focused by the fatty melon in its forehead","it sleeps with one cerebral hemisphere at a time","countercurrent heat exchangers in its flippers reduce heat loss to cold water"])
add("little brown bat","Myotis lucifugus",M,MAM,"Chiroptera","insectivore","forests near water, caves, and buildings","endothermic, with seasonal hibernation","lungs","closed, with a four-chambered heart","viviparous, with delayed fertilization through sperm storage","bilateral","urea",
 ["its wing is a patagium stretched between elongated finger bones","it locates flying insects using laryngeal echolocation","during hibernation its body temperature and heart rate drop dramatically"])
add("platypus","Ornithorhynchus anatinus",M,MAM,"Monotremata","carnivore that feeds on benthic invertebrates","freshwater streams and rivers","endothermic, with a relatively low body temperature","lungs","closed, with a four-chambered heart","oviparous, laying leathery eggs","bilateral","urea",
 ["its bill contains electroreceptors and mechanoreceptors used to locate prey underwater","males possess a venomous spur on each hind limb","females lack nipples and secrete milk through patches of skin"])
add("red kangaroo","Osphranter rufus",M,MAM,"Diprotodontia","herbivore","arid and semi-arid grassland","endothermic","lungs","closed, with a four-chambered heart","viviparous, with a short gestation followed by development in a pouch","bilateral","urea",
 ["its bipedal hopping stores elastic energy in the tendons of the hind limbs","females can exhibit embryonic diapause, pausing the development of a blastocyst","the newborn joey crawls into the pouch and attaches to a teat"])
add("giraffe","Giraffa camelopardalis",M,MAM,"Artiodactyla","herbivore that browses on tree foliage","savanna and open woodland","endothermic","lungs","closed, with a four-chambered heart","viviparous, with a gestation period of about 15 months","bilateral","urea",
 ["its neck contains seven cervical vertebrae, the same number as most mammals","a thick-walled left ventricle generates high arterial pressure to perfuse the brain","it is a ruminant with a four-compartment stomach"])
add("chimpanzee","Pan troglodytes",M,MAM,"Primates","omnivore","tropical forest and savanna woodland","endothermic","lungs","closed, with a four-chambered heart","viviparous","bilateral","urea",
 ["it manufactures and uses tools such as sticks for extracting termites","it has opposable thumbs and opposable big toes","it lives in fission-fusion communities with complex social hierarchies"])
add("polar bear","Ursus maritimus",M,MAM,"Carnivora","carnivore that feeds mainly on seals","Arctic sea ice and coastline","endothermic","lungs","closed, with a four-chambered heart","viviparous, with delayed implantation","bilateral","urea",
 ["its guard hairs are hollow and its skin beneath the fur is black","a thick subcutaneous fat layer provides insulation in cold water","it depends on sea ice as a platform for hunting seals"])
add("emperor penguin","Aptenodytes forsteri",M,AV,"Sphenisciformes","carnivore that feeds on fish, squid, and krill","Antarctic sea ice and ocean","endothermic","lungs with air sacs","closed, with a four-chambered heart","oviparous","bilateral","uric acid",
 ["its wings are modified into stiff flippers for underwater propulsion","males incubate the single egg on their feet under a brood pouch through the Antarctic winter","it huddles in large groups to reduce heat loss","it has solid, dense bones that reduce buoyancy during diving"])
add("peregrine falcon","Falco peregrinus",M,AV,"Falconiformes","carnivore that preys mainly on birds","cliffs, open landscapes, and cities","endothermic","lungs with air sacs","closed, with a four-chambered heart","oviparous","bilateral","uric acid",
 ["it reaches extremely high speeds during its hunting stoop","a notched tomial tooth on its beak helps sever the spinal cord of prey","its retina has a very high density of photoreceptors, giving it acute vision"])
add("common ostrich","Struthio camelus",M,AV,"Struthioniformes","omnivore that eats mainly plants","savanna and semi-desert","endothermic","lungs with air sacs","closed, with a four-chambered heart","oviparous, laying the largest eggs of any living bird","bilateral","uric acid",
 ["it is flightless and its sternum lacks a keel","each foot has only two toes, an adaptation for running","it is the largest living bird species"])
add("ruby-throated hummingbird","Archilochus colubris",M,AV,"Apodiformes","nectarivore that also eats small insects","woodland edges and gardens","endothermic, with nightly torpor","lungs with air sacs","closed, with a four-chambered heart","oviparous","bilateral","uric acid",
 ["it can hover by generating lift on both the downstroke and upstroke of the wing","its mass-specific metabolic rate is among the highest of any vertebrate","it migrates across the Gulf of Mexico despite its very small body mass"])
add("barn owl","Tyto alba",M,AV,"Strigiformes","carnivore that preys mainly on small rodents","farmland, grassland, and open woodland","endothermic","lungs with air sacs","closed, with a four-chambered heart","oviparous","bilateral","uric acid",
 ["its asymmetrically placed ears allow precise sound localization in the vertical plane","serrated leading edges on its flight feathers reduce flight noise","its facial disc funnels sound toward the ear openings"])
add("red junglefowl","Gallus gallus",M,AV,"Galliformes","omnivore","tropical forest edges and scrub","endothermic","lungs with air sacs","closed, with a four-chambered heart","oviparous","bilateral","uric acid",
 ["it is the wild ancestor of the domestic chicken","its developing embryo is a classic model system in developmental biology","its muscular gizzard grinds food with the aid of swallowed grit"])
add("Nile crocodile","Crocodylus niloticus",M,REP,"Crocodylia","carnivore","rivers, lakes, and marshes","ectothermic","lungs","closed, with a four-chambered heart","oviparous, with temperature-dependent sex determination","bilateral","mainly ammonia and uric acid",
 ["it has a four-chambered heart and a foramen of Panizza connecting the left and right aortas","a secondary palate allows it to breathe while its mouth is submerged","the incubation temperature of its eggs determines the sex of the hatchlings"])
add("green sea turtle","Chelonia mydas",M,REP,"Testudines","herbivore as an adult, grazing on seagrass and algae","tropical and subtropical seas","ectothermic","lungs","closed, with a three-chambered heart","oviparous, with temperature-dependent sex determination","bilateral","uric acid, urea, and ammonia",
 ["its shell is formed from the carapace and plastron fused to the ribs and vertebrae","salt glands near its eyes excrete excess sodium chloride","females return to their natal beaches to nest"])
add("green iguana","Iguana iguana",M,REP,"Squamata","herbivore","tropical forest canopy","ectothermic","lungs","closed, with a three-chambered heart","oviparous","bilateral","uric acid",
 ["it has a parietal eye on the top of its head that is sensitive to light","it uses behavioral thermoregulation, basking to raise its body temperature","it relies on hindgut fermentation by microbes to digest plant material"])
add("king cobra","Ophiophagus hannah",M,REP,"Squamata","carnivore that preys mainly on other snakes","tropical forest and mangrove swamp","ectothermic","lungs, with a reduced left lung","closed, with a three-chambered heart","oviparous, with the female building a nest","bilateral","uric acid",
 ["it injects neurotoxic venom through fixed front fangs","it detects chemical cues using its forked tongue and the vomeronasal organ","it is the longest venomous snake"])
add("Komodo dragon","Varanus komodoensis",M,REP,"Squamata","carnivore","dry grassland and tropical forest on Indonesian islands","ectothermic","lungs","closed, with a three-chambered heart","oviparous, and capable of facultative parthenogenesis","bilateral","uric acid",
 ["its mandibular glands secrete venom that disrupts blood clotting","it is the largest living lizard","females can produce offspring by parthenogenesis in the absence of males"])
add("American bullfrog","Lithobates catesbeianus",M,AMP,"Anura","carnivore as an adult","ponds, lakes, and slow streams","ectothermic","lungs, skin, and buccal lining; larvae use gills","closed, with a three-chambered heart","oviparous, with external fertilization","bilateral","urea as an adult and ammonia as a larva",
 ["its larva undergoes metamorphosis controlled by thyroid hormone","it ventilates its lungs using buccal pumping rather than a diaphragm","its moist skin serves as an important respiratory surface"])
add("axolotl","Ambystoma mexicanum",M,AMP,"Caudata","carnivore","high-altitude freshwater lakes and canals","ectothermic","external gills, lungs, and skin","closed, with a three-chambered heart","oviparous, with internal fertilization via spermatophores","bilateral","ammonia and urea",
 ["it is neotenic, retaining larval features such as external gills into sexual maturity","it can regenerate limbs, spinal cord, and parts of the heart","it is a model organism in regeneration research"])
add("cane toad","Rhinella marina",M,AMP,"Anura","omnivorous generalist predator","grassland, forest edges, and urban areas","ectothermic","lungs and skin; larvae use gills","closed, with a three-chambered heart","oviparous, with external fertilization","bilateral","urea as an adult",
 ["its parotoid glands secrete toxic bufadienolides","it became an invasive species after introduction to Australia","females can lay tens of thousands of eggs in a single clutch"])
add("great white shark","Carcharodon carcharias",M,"Chondrichthyes","Lamniformes","carnivore","temperate and subtropical coastal and oceanic waters","regionally endothermic","gills","closed, with a two-chambered heart","ovoviviparous, with oophagy by developing embryos","bilateral","urea, which is also retained in tissues for osmoregulation",
 ["its skeleton is made of cartilage rather than bone","ampullae of Lorenzini allow it to detect weak electric fields","retia mirabilia keep its swimming muscles warmer than the surrounding water","its placoid scales, or dermal denticles, reduce hydrodynamic drag"])
add("zebrafish","Danio rerio",M,"Actinopterygii","Cypriniformes","omnivore","freshwater streams and rice paddies","ectothermic","gills","closed, with a two-chambered heart","oviparous, with external fertilization","bilateral","ammonia",
 ["its transparent embryos make it a model organism for developmental biology","it can regenerate fin tissue and heart muscle","its lateral line system detects water movement"])
add("Atlantic salmon","Salmo salar",M,"Actinopterygii","Salmoniformes","carnivore","rivers and the North Atlantic Ocean","ectothermic","gills","closed, with a two-chambered heart","oviparous, with external fertilization","bilateral","ammonia",
 ["it is anadromous, migrating from the sea to freshwater to spawn","it switches its gill ion transport mechanisms when moving between freshwater and seawater","it uses olfactory cues to return to its natal stream"])
add("West Indian Ocean coelacanth","Latimeria chalumnae",M,"Sarcopterygii","Coelacanthiformes","carnivore","deep rocky slopes in the western Indian Ocean","ectothermic","gills","closed, with a two-chambered heart","ovoviviparous","bilateral","urea",
 ["it has fleshy lobed fins supported by bones homologous to tetrapod limb bones","it has an intracranial joint that allows the front of the skull to lift","it was known only from fossils until a living specimen was described in 1938"])
add("sea lamprey","Petromyzon marinus",M,"Petromyzontida","Petromyzontiformes","parasite that feeds on the blood and tissue of fish","North Atlantic and connected rivers","ectothermic","gills","closed, with a two-chambered heart","oviparous, with external fertilization","bilateral","ammonia",
 ["it is a jawless vertebrate with a round, toothed oral disc","it has a notochord that persists throughout life","its larvae, called ammocoetes, are filter feeders buried in stream sediment"])
add("big-belly seahorse","Hippocampus abdominalis",M,"Actinopterygii","Syngnathiformes","carnivore that feeds on small crustaceans","temperate seagrass beds and reefs","ectothermic","gills","closed, with a two-chambered heart","oviparous, with the male brooding eggs in a pouch","bilateral","ammonia",
 ["the male carries fertilized eggs in a specialized brood pouch until they hatch","its prehensile tail allows it to anchor to seagrass","it feeds by rapid pivot feeding, using suction to capture prey"])
I="Arthropoda"
add("western honey bee","Apis mellifera",I,"Insecta","Hymenoptera","herbivore that feeds on nectar and pollen","temperate and tropical habitats worldwide","ectothermic, with colony-level thermoregulation","a tracheal system","open, with hemolymph pumped by a dorsal vessel","oviparous, with haplodiploid sex determination","bilateral","uric acid",
 ["workers communicate the direction and distance of food sources through the waggle dance","males develop from unfertilized eggs and are haploid","the queen's pheromones regulate reproduction within the colony","it removes nitrogenous waste through Malpighian tubules"])
add("common fruit fly","Drosophila melanogaster",I,"Insecta","Diptera","saprophage that feeds on yeast in fermenting fruit","human-associated habitats worldwide","ectothermic","a tracheal system","open, with hemolymph pumped by a dorsal vessel","oviparous, with holometabolous development","bilateral","uric acid",
 ["it is a foundational model organism in genetics","its hind wings are reduced to halteres that act as gyroscopic sensors","Hox genes were first characterized through its homeotic mutations","its polytene chromosomes in larval salivary glands aided early chromosome mapping"])
add("monarch butterfly","Danaus plexippus",I,"Insecta","Lepidoptera","herbivore; larvae eat milkweed and adults feed on nectar","open fields and meadows with milkweed","ectothermic","a tracheal system","open, with hemolymph pumped by a dorsal vessel","oviparous, with complete metamorphosis","bilateral","uric acid",
 ["its larvae sequester cardenolides from milkweed, making them distasteful to predators","eastern North American populations undertake a multigenerational migration to Mexico","it uses a time-compensated sun compass for orientation"])
add("desert locust","Schistocerca gregaria",I,"Insecta","Orthoptera","herbivore","arid regions of Africa, the Middle East, and South Asia","ectothermic","a tracheal system","open, with hemolymph pumped by a dorsal vessel","oviparous, with incomplete metamorphosis","bilateral","uric acid",
 ["crowding triggers a switch from a solitary to a gregarious phase","serotonin plays a key role in the transition to gregarious behavior","gregarious swarms can migrate long distances and devastate crops"])
add("southern black widow","Latrodectus mactans",I,"Arachnida","Araneae","carnivore","sheltered sites in temperate regions of North America","ectothermic","book lungs and tracheae","open, with hemolymph containing hemocyanin","oviparous","bilateral","guanine and uric acid",
 ["its venom contains alpha-latrotoxin, which triggers massive neurotransmitter release","it produces silk from abdominal spinnerets","it digests prey externally by injecting digestive enzymes"])
add("American lobster","Homarus americanus",I,"Malacostraca","Decapoda","omnivorous scavenger and predator","rocky and muddy seafloor of the northwest Atlantic","ectothermic","gills","open, with hemolymph containing hemocyanin","oviparous, with females carrying eggs on their pleopods","bilateral","ammonia",
 ["it grows by molting its chitinous exoskeleton","its two claws are specialized differently, one for crushing and one for cutting","it can regenerate lost limbs over successive molts"])
add("Atlantic horseshoe crab","Limulus polyphemus",I,"Merostomata","Xiphosura","carnivore that feeds on benthic invertebrates","shallow coastal waters of the western Atlantic","ectothermic","book gills","open, with hemolymph containing hemocyanin","oviparous, with external fertilization","bilateral","ammonia",
 ["its amebocytes are the source of Limulus amebocyte lysate used to detect bacterial endotoxins","its lateral compound eyes have been a model system in visual neuroscience","despite its name it is more closely related to arachnids than to crabs"])
MO="Mollusca"
add("common octopus","Octopus vulgaris",MO,"Cephalopoda","Octopoda","carnivore","coastal rocky seafloor","ectothermic","gills","closed, with three hearts","oviparous, with semelparous reproduction","bilateral","ammonia",
 ["it has two branchial hearts and one systemic heart","its blood contains copper-based hemocyanin","chromatophores controlled by muscles allow rapid color change","a large proportion of its neurons are located in its arms"])
add("giant squid","Architeuthis dux",MO,"Cephalopoda","Oegopsida","carnivore","deep ocean","ectothermic","gills","closed, with three hearts","oviparous","bilateral","ammonia",
 ["it has among the largest eyes of any animal","it moves by jet propulsion, expelling water through a muscular siphon","its internal shell is reduced to a chitinous gladius"])
add("garden snail","Cornu aspersum",MO,"Gastropoda","Stylommatophora","herbivore","gardens, hedgerows, and woodland","ectothermic","a pallial lung formed from the mantle cavity","open, with hemolymph containing hemocyanin","oviparous, as a simultaneous hermaphrodite","asymmetrical due to torsion and coiling","ammonia and uric acid",
 ["it feeds by rasping plant material with a radula","its body undergoes torsion during larval development","it can aestivate by sealing its shell opening with a dried mucus epiphragm"])
add("blue mussel","Mytilus edulis",MO,"Bivalvia","Mytilida","suspension feeder","intertidal and subtidal rocky shores","ectothermic","gills","open","oviparous, with external fertilization","bilateral","ammonia",
 ["it attaches to rocks using protein-based byssal threads","its gills function both in respiration and in trapping food particles","it lacks a distinct head and radula"])
add("common earthworm","Lumbricus terrestris",I.replace("Arthropoda","Annelida"),"Clitellata","Crassiclitellata","detritivore","moist soil","ectothermic","the moist body surface","closed, with five pairs of aortic arches","oviparous, as a simultaneous hermaphrodite that cross-fertilizes","bilateral","ammonia and urea",
 ["its body is divided into repeated segments called metameres","its hydrostatic skeleton works with circular and longitudinal muscles to produce peristaltic movement","each segment contains a pair of metanephridia for excretion","it deposits eggs in a cocoon secreted by the clitellum"])
add("medicinal leech","Hirudo medicinalis","Annelida","Clitellata","Arhynchobdellida","hematophage that feeds on vertebrate blood","freshwater ponds and marshes","ectothermic","the body surface","closed","oviparous, as a hermaphrodite","bilateral","ammonia",
 ["its saliva contains hirudin, an anticoagulant that inhibits thrombin","it attaches using anterior and posterior suckers","it has been used as a model for studying simple neural circuits"])
C="Cnidaria"
add("moon jellyfish","Aurelia aurita",C,"Scyphozoa","Semaeostomeae","carnivore that feeds on plankton","coastal marine waters","ectothermic","diffusion across the body surface","absent, with a gastrovascular cavity distributing nutrients","alternating between sexual medusae and asexual polyps","radial","ammonia",
 ["it is diploblastic, with an ectoderm and endoderm separated by mesoglea","its cnidocytes contain stinging organelles called nematocysts","its life cycle includes a sessile polyp stage that buds off ephyrae by strobilation"])
add("staghorn coral","Acropora cervicornis",C,"Anthozoa","Scleractinia","mixotroph that captures plankton and receives nutrients from symbiotic algae","shallow tropical reefs","ectothermic","diffusion across the body surface","absent, with a gastrovascular cavity","both sexual broadcast spawning and asexual fragmentation","radial","ammonia",
 ["it hosts photosynthetic dinoflagellate symbionts within its gastrodermal cells","it secretes a calcium carbonate skeleton made of aragonite","coral bleaching occurs when heat stress disrupts its symbiosis with algae"])
add("green hydra","Hydra viridissima",C,"Hydrozoa","Anthoathecata","carnivore","freshwater ponds and lakes","ectothermic","diffusion across the body surface","absent, with a gastrovascular cavity","mainly asexual budding, with sexual reproduction under stress","radial","ammonia",
 ["it has a decentralized nerve net rather than a centralized brain","its body is continuously renewed by stem cells, giving it remarkable regenerative capacity","it harbors symbiotic green algae in its endodermal cells"])
E="Echinodermata"
add("common starfish","Asterias rubens",E,"Asteroidea","Forcipulatida","carnivore that preys mainly on bivalves","rocky and sandy seafloor of the northeast Atlantic","ectothermic","tube feet and dermal branchiae","a water vascular system and a reduced hemal system","oviparous, with external fertilization and a bilateral larva","pentaradial as an adult and bilateral as a larva","ammonia",
 ["it moves using hydraulically operated tube feet of the water vascular system","it can evert its stomach to digest prey outside its body","it can regenerate lost arms","its endoskeleton consists of calcareous ossicles"])
add("purple sea urchin","Strongylocentrotus purpuratus",E,"Echinoidea","Camarodonta","herbivore that grazes on kelp","rocky intertidal and subtidal zones of the eastern Pacific","ectothermic","tube feet and peristomial gills","a water vascular system","oviparous, with external fertilization","pentaradial as an adult and bilateral as a larva","ammonia",
 ["it scrapes algae using a jaw apparatus called Aristotle's lantern","its gametes have been central to studies of fertilization","it is a deuterostome, like chordates"])
add("Venus' flower basket","Euplectella aspergillum","Porifera","Hexactinellida","Lyssacinosida","filter feeder","deep ocean seafloor","ectothermic","diffusion driven by water currents","absent, with water flowing through a canal system","both sexual and asexual reproduction","asymmetrical","ammonia",
 ["it lacks true tissues and organs","its skeleton is built from siliceous spicules fused into a lattice","choanocytes with flagella drive water through its body"])
add("Caenorhabditis elegans","Caenorhabditis elegans","Nematoda","Chromadorea","Rhabditida","bacterivore","soil and rotting vegetation","ectothermic","diffusion across the cuticle","absent, with fluid in the pseudocoelom","mainly self-fertilizing hermaphrodites with rare males","bilateral","ammonia",
 ["its complete cell lineage from zygote to adult has been mapped","the adult hermaphrodite has a fixed number of somatic cells","it was the first multicellular organism to have its genome sequenced","its connectome of 302 neurons in the hermaphrodite has been mapped"])
add("freshwater planarian","Schmidtea mediterranea","Platyhelminthes","Rhabditophora","Tricladida","carnivore","freshwater streams and ponds","ectothermic","diffusion across the body surface","absent","sexual as a hermaphrodite or asexual by fission, depending on the strain","bilateral","ammonia",
 ["it is acoelomate, lacking a body cavity between gut and body wall","pluripotent stem cells called neoblasts enable whole-body regeneration","it excretes excess water through protonephridia with flame cells"])
add("tardigrade","Hypsibius exemplaris","Tardigrada","Eutardigrada","Parachela","herbivore that pierces algal cells","freshwater films on mosses and sediments","ectothermic","diffusion across the body surface","open, with fluid in the body cavity","parthenogenetic","bilateral","ammonia",
 ["it can enter cryptobiosis, surviving extreme desiccation","it expresses intrinsically disordered proteins that help protect cells during drying","it has four pairs of lobopod legs ending in claws"])

import re
for a in A:
    a["dshort"]=re.split(r" that |;|,| as an adult| as a larva",a["diet"])[0].replace("omnivorous generalist predator","generalist predator").replace("omnivorous scavenger and predator","scavenger and predator")
    a["cshort"]=a["circ"].split(",")[0]
    a["tshort"]=a["thermo"].split(",")[0]
    a["wshort"]=a["waste"].replace("mainly ","").split(",")[0]
    a["rshort"]=a["resp"].split(";")[0].split(", with")[0]
vertebrate_classes={"Mammalia","Aves","Reptilia","Amphibia","Chondrichthyes","Actinopterygii","Sarcopterygii","Petromyzontida"}
S=[]
def put(cat,s):
    s=s.strip(); s=s[0].upper()+s[1:]
    if not s.endswith("."): s+="."
    S.append((cat,s))

def art(w): return "an" if w[0].lower() in "aeiou" else "a"
for a in A:
    n,sci=a["name"],a["sci"]
    The = n if n[0].isupper() and n!="Caenorhabditis elegans" and n.split()[0] not in ("American","Atlantic","African","Nile","West","Komodo","Venus'") else "the "+n
    if n=="Caenorhabditis elegans": The="the nematode Caenorhabditis elegans"
    Th = The
    put("taxonomy",f"The scientific name of {Th} is {sci}")
    put("taxonomy",f"{sci} is the binomial name for {Th}")
    put("taxonomy",f"{Th.capitalize() if Th.startswith('the') else Th} belongs to the phylum {a['phylum']}")
    put("taxonomy",f"{sci} is classified in the class {a['cls']}")
    put("taxonomy",f"{sci} is placed in the order {a['order']}")
    put("taxonomy",f"Taxonomically, {Th} ({sci}) is a member of the phylum {a['phylum']}, class {a['cls']}, and order {a['order']}")
    put("taxonomy",f"Within the class {a['cls']}, {sci} belongs to the order {a['order']}")
    if a["cls"] in vertebrate_classes: put("taxonomy",f"{Th} is a vertebrate, possessing a vertebral column or its cartilaginous equivalent" if a["cls"]!="Petromyzontida" else f"{Th} is a jawless vertebrate")
    else: put("taxonomy",f"{Th} is an invertebrate, lacking a vertebral column")
    put("ecology",f"{Th} is {art(a['diet'])} {a['diet']}")
    put("ecology",f"In terms of trophic ecology, {sci} is {art(a['diet'])} {a['diet']}")
    put("ecology",f"{Th} inhabits {a['habitat']}")
    put("ecology",f"The typical habitat of {sci} is {a['habitat']}")
    put("ecology",f"{sci} is {art(a['diet'])} {a['diet']} found in {a['habitat']}")
    put("physiology",f"{Th} is {a['thermo']}")
    put("physiology",f"With respect to thermoregulation, {sci} is {a['thermo']}")
    put("physiology",f"Gas exchange in {Th} occurs through {a['resp']}")
    put("physiology",f"{sci} respires using {a['resp']}")
    put("physiology",f"The circulatory system of {Th} is {a['circ']}")
    put("physiology",f"In {sci}, circulation is {a['circ']}")
    put("physiology",f"The main nitrogenous waste excreted by {Th} is {a['waste']}")
    put("physiology",f"{sci} eliminates nitrogen primarily as {a['waste']}")
    put("reproduction",f"Reproduction in {Th} is {a['repro']}")
    put("reproduction",f"The reproductive mode of {sci} can be described as {a['repro']}")
    put("morphology",f"{Th} has {a['sym']} body symmetry")
    put("morphology",f"The body plan of {sci} shows {a['sym']} symmetry")
    for t in a["traits"]:
        put("traits",f"Regarding {Th}, {t}")
        put("traits",f"A notable feature of {Th} is that {t}")
        put("traits",f"Researchers studying {sci} have noted that {t}")
    put("summary",f"{sci}, commonly known as {Th}, is {art(a['dshort'])} {a['dshort']} of the class {a['cls']} that respires using {a['resp']}")
    put("summary",f"{Th} is a {a['thermo'].split(',')[0]} member of the class {a['cls']} whose circulatory system is {a['circ']}")

# comparisons
comp=[]
T={
 "tshort":["Unlike {y}, which is {Y}, {x} is {X}","{x} is {X}, whereas {y} is {Y}","In contrast to the {Y} {y}, {x} is {X}"],
 "dshort":["Unlike {y}, which is {aY} {Y}, {x} is {aX} {X}","{x} is {aX} {X}, whereas {y} is {aY} {Y}","While {y} is {aY} {Y}, {x} is {aX} {X}"],
 "wshort":["{x} excretes nitrogen mainly as {X}, whereas {y} excretes it mainly as {Y}","Unlike {y}, which excretes {Y}, {x} excretes {X}"],
 "rshort":["{x} exchanges gases through {X}, whereas {y} uses {Y}","Unlike {y}, which respires through {Y}, {x} respires through {X}"],
 "sym":["{x} has {X} symmetry, whereas {y} has {Y} symmetry"],
 "cshort":["{x} has a circulatory system that is {X}, whereas that of {y} is {Y}","Unlike {y}, whose circulatory system is {Y}, {x} has a circulatory system that is {X}"],
}
for x,y in itertools.permutations(A,2):
    for k,ts in T.items():
        if x[k]!=y[k] and not (k=="cshort" and "absent" in (x[k],y[k])):
            for t in ts:
                comp.append(("comparative",t.format(x=x["sci"],y=y["sci"],X=x[k],Y=y[k],aX=art(x[k]),aY=art(y[k]))))
    if x["cls"]==y["cls"] and x["sci"]<y["sci"]:
        comp.append(("comparative",f"{x['sci']} and {y['sci']} both belong to the class {x['cls']}"))
        comp.append(("comparative",f"Both {x['sci']} and {y['sci']} are members of the order {x['order']}" if x['order']==y['order'] else f"Although {x['sci']} and {y['sci']} share the class {x['cls']}, they are placed in different orders, {x['order']} and {y['order']}"))
    if x["phylum"]!=y["phylum"] and x["sci"]<y["sci"]:
        comp.append(("comparative",f"{x['sci']} belongs to the phylum {x['phylum']}, while {y['sci']} belongs to the phylum {y['phylum']}"))

concepts = """Endothermic animals generate most of their body heat through metabolic processes.
Ectothermic animals rely mainly on external heat sources to regulate body temperature.
Homeostasis is the maintenance of a relatively stable internal environment despite external changes.
Negative feedback loops counteract deviations from a physiological set point.
Positive feedback loops amplify a physiological response, as occurs during childbirth in mammals.
The surface-area-to-volume ratio decreases as body size increases, which affects heat exchange.
Basal metabolic rate scales with body mass raised to an exponent close to three quarters, a relationship known as Kleiber's law.
Countercurrent exchange maximizes the transfer of heat or solutes between fluids flowing in opposite directions.
Fish gills use countercurrent exchange to extract oxygen efficiently from water.
Bird lungs are ventilated unidirectionally by a system of air sacs.
Mammalian lungs are ventilated tidally, with air moving in and out through the same airways.
The diaphragm is a sheet of skeletal muscle that drives inspiration in mammals.
Insects deliver oxygen directly to tissues through a branching network of tracheae opening at spiracles.
Hemoglobin is an iron-containing respiratory pigment found in the blood of vertebrates and some invertebrates.
Hemocyanin is a copper-containing respiratory pigment found in many molluscs and arthropods.
Myoglobin stores oxygen in muscle tissue and is especially abundant in diving mammals.
The Bohr effect describes the decrease in hemoglobin's oxygen affinity at lower pH.
In an open circulatory system, hemolymph bathes the organs directly within a hemocoel.
In a closed circulatory system, blood remains confined within vessels.
Double circulation separates the pulmonary and systemic circuits.
The four-chambered heart of birds and mammals completely separates oxygenated and deoxygenated blood.
The sinoatrial node acts as the pacemaker of the vertebrate heart.
Ammonia is highly toxic and is usually excreted by aquatic animals that can dilute it in large volumes of water.
Urea is less toxic than ammonia and is synthesized in the liver of mammals through the ornithine cycle.
Uric acid is excreted as a paste with little water, conserving water in birds, reptiles, and insects.
The nephron is the functional unit of the vertebrate kidney.
The loop of Henle creates a concentration gradient that allows mammals to produce hyperosmotic urine.
Malpighian tubules remove nitrogenous wastes from the hemolymph of insects.
Osmoconformers maintain internal osmolarity similar to that of their environment.
Osmoregulators actively control internal osmolarity regardless of external conditions.
Marine bony fishes drink seawater and excrete excess salts through their gills.
Freshwater fishes produce large volumes of dilute urine and take up ions through their gills.
Neurons transmit electrical signals called action potentials along their axons.
Myelin increases the conduction velocity of action potentials in vertebrate axons.
Synapses transmit signals between neurons through chemical neurotransmitters or electrical junctions.
The cerebellum coordinates motor control and balance in vertebrates.
Cephalization is the concentration of sensory organs and nervous tissue at the anterior end of the body.
Radially symmetrical animals often have a diffuse nerve net rather than a centralized brain.
Compound eyes consist of many ommatidia, each with its own lens.
Camera-type eyes evolved independently in vertebrates and cephalopods.
Electroreception allows some aquatic animals to detect weak electric fields generated by other organisms.
Echolocation involves emitting sounds and interpreting the returning echoes to locate objects.
The lateral line system of fishes and aquatic amphibians detects water movements and pressure changes.
Hormones are chemical messengers secreted into the circulation that act on distant target cells.
The hypothalamus links the nervous system to the endocrine system through the pituitary gland.
Ecdysone is a steroid hormone that triggers molting in arthropods.
Juvenile hormone maintains larval characteristics in insects and regulates metamorphosis.
Thyroid hormones control metamorphosis in amphibians.
Insulin lowers blood glucose by promoting cellular uptake and glycogen synthesis.
Chitin is a nitrogen-containing polysaccharide that forms the exoskeleton of arthropods.
Molting, or ecdysis, allows arthropods to grow despite their rigid exoskeleton.
A hydrostatic skeleton consists of a fluid-filled cavity surrounded by muscles.
Endoskeletons are internal support structures made of bone, cartilage, or calcareous ossicles.
Skeletal muscle contraction is explained by the sliding filament model.
Calcium ions bind to troponin, allowing myosin to interact with actin during muscle contraction.
Asynchronous flight muscles allow some insects to beat their wings faster than their nerve impulses fire.
Diploblastic animals develop from two germ layers, the ectoderm and the endoderm.
Triploblastic animals develop from three germ layers, including a mesoderm.
A coelom is a body cavity fully lined by tissue derived from mesoderm.
Acoelomate animals lack a body cavity between the gut and the body wall.
Pseudocoelomate animals have a body cavity that is only partially lined by mesoderm.
In protostomes, the blastopore typically develops into the mouth.
In deuterostomes, the blastopore typically develops into the anus.
Echinoderms and chordates are both deuterostomes.
Gastrulation is the developmental process that establishes the germ layers.
Hox genes specify positional identity along the anterior-posterior axis of animal embryos.
Neural crest cells are a vertebrate innovation that give rise to many cell types, including pigment cells and parts of the skull.
All chordates possess a notochord, a dorsal hollow nerve cord, pharyngeal slits, and a post-anal tail at some stage of development.
The amniotic egg allowed reptiles, birds, and mammals to reproduce away from water.
The amnion, chorion, allantois, and yolk sac are extraembryonic membranes of the amniotic egg.
Oviparous animals lay eggs that develop outside the mother's body.
Viviparous animals give birth to live young that develop inside the mother's body.
Ovoviviparous animals retain eggs within the body until they hatch.
Parthenogenesis is the development of an embryo from an unfertilized egg.
Hermaphroditic animals possess both male and female reproductive organs.
Sequential hermaphrodites change sex during their lifetime.
External fertilization occurs when gametes are released into the environment.
Internal fertilization occurs when sperm are delivered into the female reproductive tract.
Complete metamorphosis in insects includes egg, larva, pupa, and adult stages.
Incomplete metamorphosis in insects proceeds through egg, nymph, and adult stages.
Temperature-dependent sex determination occurs in many reptiles.
Haplodiploidy is a sex-determination system in which males are haploid and females are diploid.
Semelparous animals reproduce once before dying, whereas iteroparous animals reproduce repeatedly.
Ruminants digest cellulose with the help of symbiotic microorganisms in a multi-chambered stomach.
Hindgut fermenters digest plant material with microbes in the cecum or colon.
Carnivores generally have shorter digestive tracts than herbivores of similar body size.
Filter feeders strain small food particles suspended in water.
The radula is a tongue-like feeding structure unique to molluscs.
Enzymes such as pepsin, trypsin, and lipase break down macromolecules during digestion.
Biological classification ranks organisms in hierarchical categories such as phylum, class, order, family, genus, and species.
The binomial nomenclature system assigns each species a genus name and a specific epithet.
A clade consists of an ancestor and all of its descendants.
Phylogenetic trees represent hypotheses about evolutionary relationships among organisms.
Convergent evolution produces similar traits in unrelated lineages facing similar selective pressures.
Homologous structures share a common evolutionary origin, even if their functions differ.
Analogous structures perform similar functions but evolved independently.
Sponges are generally considered among the earliest-diverging animal lineages.
Cnidarians possess stinging cells called cnidocytes.
Arthropods are the most species-rich animal phylum.
Molluscs typically have a mantle, a muscular foot, and a visceral mass.
Annelids are segmented worms with a true coelom.
Nematodes are covered by a cuticle that they molt as they grow.
Torpor is a short-term reduction in metabolic rate and body temperature.
Hibernation is a prolonged state of torpor that allows animals to survive cold seasons with scarce food.
Aestivation is a state of dormancy that helps animals survive hot or dry periods.
Cryptobiosis is a reversible state in which metabolic activity becomes undetectable.
Brown adipose tissue produces heat through non-shivering thermogenesis.
Uncoupling protein 1 allows mitochondria in brown fat to release energy as heat rather than producing ATP.
Blubber is a thick layer of subcutaneous fat that insulates marine mammals.
Innate immunity provides rapid, nonspecific defense against pathogens.
Adaptive immunity, based on lymphocytes with specific receptors, is found in vertebrates.
Jawless vertebrates use variable lymphocyte receptors rather than antibodies for adaptive immunity.
Circadian rhythms are endogenous cycles of approximately 24 hours.
Migration is a seasonal, often long-distance movement of animals between habitats.
Many migratory birds use the Earth's magnetic field for navigation.
Kin selection can explain the evolution of altruistic behavior among relatives.
Eusocial animals show reproductive division of labor, overlapping generations, and cooperative brood care.
Mimicry occurs when one species evolves to resemble another.
Aposematic coloration warns predators that an animal is toxic or distasteful.
Sexual selection favors traits that increase mating success.
Model organisms are species that are extensively studied to understand general biological principles.
Regeneration is the ability to regrow lost or damaged tissues and body parts.""".strip().split("\n")
prefixes=["","In animal biology, ","In comparative physiology, ","From a zoological perspective, ","As a general principle, "]
KEEP={"Kleiber's","Hox","Malpighian"}
for c in concepts:
    low = c if c.split()[0] in KEEP else c[0].lower()+c[1:]
    for p in prefixes:
        put("concepts", p+low if p else c)

# dedupe
seen=set(); base=[]
for cat,s in S:
    if s not in seen: seen.add(s); base.append((cat,s))
random.shuffle(comp)
comps=[]
for cat,s in comp:
    s=s[0].upper()+s[1:]+"."
    if s not in seen: seen.add(s); comps.append((cat,s))
need=10000-len(base)
rows=base+comps[:need]
random.shuffle(rows)
print(len(base),len(comps),len(rows))
with open(__import__("pathlib").Path(__file__).with_name("animal_biology_sentences.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["id","category","sentence"])
    for i,(c,s) in enumerate(rows,1): w.writerow([i,c,s])
