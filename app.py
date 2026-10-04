"""Analyseur d'achat Vinted : Streamlit + API Google Gemini (offre gratuite, texte + vision).

Local : pip install -r requirements.txt, puis : streamlit run app.py
Clé API : variable d'environnement GEMINI_API_KEY, ou .streamlit/secrets.toml,
          ou « Secrets » de Streamlit Community Cloud.
Optionnel : GEMINI_MODEL (impose un modèle), APP_PASSWORD (protège l'accès à l'app).
"""
import hmac
import io
import os
from datetime import date
from urllib.parse import quote_plus

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

st.set_page_config(page_title="Analyseur d'achat Vinted", page_icon="🛍️", layout="centered")

# --- Réglages -----------------------------------------------------------------
# Modèles gratuits essayés dans l'ordre : le suivant prend le relais si le quota
# est atteint ou si un modèle est retiré par Google.
MODELES = [m for m in (os.getenv("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash") if m]
ACHAT_MINI = 0.5  # € : évite une division par zéro
ETATS = [
    "Neuf avec étiquette",
    "Neuf sans étiquette",
    "Très bon état",
    "Bon état",
    "Satisfaisant (usé, défauts visibles)",
    "Testé, fonctionne (électronique)",
    "Non testé / état inconnu",
    "Pour pièces / en panne",
]

# --- Logique du guide, injectée dans le prompt système de l'IA ------------------
SYSTEM_PROMPT = """Tu es l'analyste d'achat d'un revendeur débutant en France : il achète en brocantes permanentes (Emmaüs, friperies solidaires, ressourceries, dépôts-ventes, vide-greniers) pour revendre sur Vinted. Date du jour : {aujourdhui}.
Pour chaque article (texte + photo éventuelle), tu décides s'il faut l'ACHETER pour le revendre et tu chiffres le potentiel en appliquant STRICTEMENT le guide ci-dessous. Sois prudent : un mauvais achat coûte plus cher qu'une occasion ratée.

## Calcul et règles d'achat
- Net par pièce = prix encaissé × 0,877 − achat − emballage (0,877 = 12,3 % de cotisations micro-entrepreneur). Emballage : 0,5 € pour un vêtement, 1 à 2 € pour une boîte.
- Vinted ne prélève aucune commission au vendeur ; l'acheteur paie la protection (0,70 € + 5 %) et le port, donc il compare le prix total.
- ACHETER seulement si la revente médiane vaut au moins 4 fois le prix d'achat (ou le seuil donné par l'utilisateur), avec au moins 3 annonces comparables probables, et si le net reste positif. Une seule pièce non testée à la fois, 8 € maximum. Jamais plus de 25 % du budget sur une catégorie.
- prix_revente = prix réellement encaissé sur Vinted France (les offres et les lots font baisser le prix affiché), pas un prix d'annonce gonflé. Tu n'as pas les annonces en direct : donne une confiance honnête.

## Catalogue : achat typique → revente médiane [fourchette] ; part vendue en 30 jours
Valeurs sûres (groupe S) :
- Polaires et vestes outdoor de marque (Patagonia, The North Face, Columbia) : 7 € → 38 € [20–70], 70–80 %. Pic de septembre à février. Risques : faux The North Face/Arc'teryx, zip cassé, odeurs. Contrôle : étiquette avec numéro de style (« STY » chez Patagonia), logo brodé net, coutures régulières.
- Marques FR premium femme (Sézane, Maje, Sandro, Claudie Pierlot, Zadig & Voltaire) : 9 € → 45 € [25–110], 65–75 %. Risques : faux (Zadig & Voltaire, Isabel Marant), pilling. Contrôle : étiquette propre, composition et « Made in » cohérents, doublure soignée ; privilégier laine, cachemire, soie, lin.
- Denim et workwear (Levi's, Carhartt, Dickies) : 6 € → 30 € [18–55], 65–75 %. Préciser coupe et W/L ; Carhartt WIP se vend plus cher. Contrôle : étiquette rouge Levi's, « Made in USA », entrejambe et ourlets usés, faux Carhartt.
- Appareils photo compacts 2000s (Canon IXUS, Sony Cyber-shot) : 12 € → 55 € [30–120], 55–65 %. Uniquement testés (batterie + carte) ; sans test 8–10 € maximum. Contrôle : objectif, flash, écran, zoom.
- Pulls laine / cachemire de marque : 5 € → 28 € [15–60], 65–75 %, forte d'octobre à février. Lire la composition (100 % cachemire, mérinos, lambswool) ; refuser acrylique, trous, mites, odeurs.
- Sportswear vintage (Adidas, Nike, Fila, Ellesse, Lacoste, Kappa, Champion) : 5 € → 25 € [12–60], 55–70 %. Vestes de survêtement des années 80–90 ; étiquettes d'époque, broderies nettes ; risque de faux Nike/Adidas.
- LEGO et Playmobil : lot de 5–15 € → 32 € [15–60], 55–65 %. Sets identifiables (numéro, notice), minifigs de licences ; vrac au-dessus de 8 €/kg : la marge disparaît ; sets incomplets et clones.
- Bijoux argent 925 / Swarovski : 4 € → 20 € [10–60], 45–60 %. Poinçon 925 ou tête de Minerve, test à l'aimant ; plaqué vendu comme argent.
- Mangas (lot de 10 tomes) : 10–20 € → 45 € [25–90], 55–65 %. Séries complètes et récentes ; port acheteur 5–7 € ; tomes manquants.
- Jeux de société complets : 4 € → 18 € [10–35], 40–55 %. Compter les pièces ; jeux modernes connus (Catan, Carcassonne, Ticket to Ride, Dixit, Azul).
Pépites rares (groupe P) :
- Maillots de foot rétro authentiques (1985–2008) : 7 € → 40 € [20–120+], 50–60 %. Fort risque de faux : étiquettes d'époque cohérentes, écusson et sponsor brodés, flocage d'origine ; un maillot « trop parfait » ou trop peu cher est suspect. Demande en creux d'octobre à janvier (après la Coupe du monde 2026).
- Consoles et jeux rétro (Game Boy, GBC, GBA, DS Lite, PSP) : 20 € → 60 € [40–90], jeux 12–40 €, 60–70 %. Repères : DS Lite ≈ 54 €, GBA ≈ 60 €, GBC 75–90 €. Test obligatoire ; cartouches : étiquette nette, circuit propre, méfiance envers un Pokémon bon marché ; pic à Noël.
- Cartes à collectionner (Pokémon, Magic, Yu-Gi-Oh) : 0,2–1 € la carte → 1–5 € (communes), 20–200 €+ (rares). Bonus, pas un plan ; vérifier chaque carte sur Cardmarket ; faux fréquents.
- Montres Seiko / Citizen / Casio : 15 € → 60 € [30–150], 40–55 %, rotation lente. Contrôle : référence au dos, trotteuse qui avance, couronne et verre en bon état ; viser Seiko 5 et Citizen automatiques vintage.
À éviter (groupe X, NE PAS ACHETER) : fast-fashion (Zara, H&M, Shein, Primark : 3 € → 9 €, net ≈ 4 €) ; livres de poche, DVD, CD ; vinyles courants (sauf si Discogs affiche au moins 25 €) ; vaisselle lourde, fonte, électroménager, meubles (port 6–12 €, casse) ; électronique non testée en vrac (25–40 % de pannes) ; sacs et bijoux « de luxe » non authentifiables (retrait d'annonce, délit de contrefaçon) ; chaussures usées, sous-vêtements.

## Article hors catalogue (groupe H)
Raisonne par analogie avec les catégories ci-dessus : demande et rareté, marque, prix probables d'annonces comparables, poids et port, risque de faux ou de panne, besoin de test, saisonnalité à la date du jour (polaires, pulls, manteaux : pic de septembre à février). Si tu ne peux pas justifier un multiplicateur d'au moins 4 par des raisons concrètes, réponds NE PAS ACHETER.

## Effet de l'état
Neuf avec étiquette : haut de la fourchette. Très bon état : milieu haut. Bon état : milieu. Satisfaisant : bas de la fourchette (−30 à −50 %). Non testé : électronique à 8 € maximum, risque élevé. Pour pièces ou en panne : NE PAS ACHETER, sauf pièce de collection évidente.

## Photo
Si une photo est jointe, identifie marque, modèle, étiquettes lisibles, matière, état réel (taches, trous, usure, boulochage) et indices de contrefaçon. Si elle contredit la saisie, fie-toi à la photo et dis-le. Contrefaçon probable : NE PAS ACHETER. Sans photo, baisse la confiance.

## Réponse
Réponds uniquement avec le JSON demandé, en français, prix en euros. Si l'utilisateur donne un prix d'achat, juge CE prix ; sinon estime prix_achat_typique. La justification tient en 3 phrases courtes au maximum (pourquoi ce prix de revente, risque principal, condition d'achat) et ne cite aucun multiplicateur chiffré : l'application le calcule.
"""


class Analyse(BaseModel):
    categorie: str = Field(description="Catégorie du guide la plus proche, ou type d'article si hors catalogue")
    groupe: str = Field(description="S = valeur sûre, P = pépite rare, X = piège à éviter, H = hors catalogue")
    decision: str = Field(description="ACHETER ou NE PAS ACHETER")
    prix_revente: float = Field(description="Prix de revente médian réaliste encaissé sur Vinted France, en euros")
    prix_revente_min: float = Field(description="Bas de la fourchette de revente, en euros")
    prix_revente_max: float = Field(description="Haut de la fourchette de revente, en euros")
    prix_achat_typique: float = Field(description="Prix payé typiquement pour cet article en brocante, en euros")
    emballage: float = Field(description="Coût d'emballage en euros : 0.5 pour un vêtement, 1 à 2 pour une boîte")
    delai_vente_jours: int = Field(description="Délai de vente estimé sur Vinted, en jours")
    risque: str = Field(description="faible, moyen ou élevé (faux, panne, défauts)")
    confiance: str = Field(description="faible, moyenne ou élevée")
    justification: list[str] = Field(description="3 phrases courtes maximum, sans multiplicateur chiffré")
    a_verifier: str = Field(description="Le contrôle le plus important à faire en rayon avant d'acheter")


# --- Fonctions ------------------------------------------------------------------
def lire_secret(nom):
    """Variable d'environnement, sinon st.secrets (secrets.toml ou Secrets de Community Cloud)."""
    valeur = os.getenv(nom)
    if valeur:
        return valeur
    try:
        return st.secrets.get(nom)
    except Exception:
        return None


def fr(x, n=1):
    """Format français : 1 234,5"""
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",")


def md(texte):
    """Évite que Streamlit interprète les $ comme du LaTeX."""
    return str(texte).replace("$", "\\$")


def preparer_image(source):
    """Corrige l'orientation, réduit à 1280 px et convertit en JPEG (économise le quota)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(source.getvalue()))).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def construire_demande(nom, nature, etat, prix, seuil, avec_photo):
    return "\n".join([
        f"Nom exact de l'article : {nom}",
        f"Nature / description : {nature or 'non précisée'}",
        f"État déclaré : {etat}",
        f"Prix d'achat demandé : {prix:.2f} € (juge CE prix)" if prix > 0
        else "Prix d'achat : non renseigné (estime le prix d'achat typique en brocante).",
        f"Multiplicateur minimum exigé par l'utilisateur : ×{seuil:g}",
        "Une photo est jointe." if avec_photo else "Aucune photo fournie.",
    ])


def analyser(cle, demande, jpeg):
    client = genai.Client(api_key=cle, http_options=types.HttpOptions(timeout=60_000))
    contenu = ([types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")] if jpeg else []) + [demande]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT.replace("{aujourdhui}", date.today().strftime("%d/%m/%Y")),
        response_mime_type="application/json",
        response_schema=Analyse,
    )
    derniere = None
    for modele in MODELES:
        try:
            rep = client.models.generate_content(model=modele, contents=contenu, config=config)
            analyse = rep.parsed if isinstance(rep.parsed, Analyse) else Analyse.model_validate_json(rep.text)
            return analyse, modele
        except Exception as e:  # quota (429), modèle retiré (404), surcharge (503), JSON invalide
            derniere = e
            reessayer = isinstance(e, ValueError) or any(
                c in str(e) for c in ("404", "429", "500", "503", "NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
            )
            if not reessayer:
                raise
    raise derniere


def expliquer(e):
    m = str(e)
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "Quota gratuit atteint sur les modèles essayés. Réessaie dans une minute, ou demain si c'est la limite journalière."
    if "API_KEY_INVALID" in m or "API key not valid" in m:
        return "Clé API refusée : vérifie GEMINI_API_KEY (à créer sur aistudio.google.com/apikey)."
    if "404" in m or "NOT_FOUND" in m:
        return "Aucun modèle disponible. Renseigne GEMINI_MODEL avec un modèle gratuit listé sur ai.google.dev/gemini-api/docs/models."
    return f"Erreur pendant l'analyse : {m[:300]}"


def afficher_resultat(r):
    a, saisi, seuil, taux = r["a"], r["saisi"], r["seuil"], r["taux"]
    # Le multiplicateur, le net et la décision finale sont recalculés ici : on ne dépend pas de l'arithmétique du modèle.
    achat = max(saisi if saisi > 0 else a.prix_achat_typique, ACHAT_MINI)
    revente = max(a.prix_revente, 0.0)
    mult = revente / achat
    net = revente * (1 - taux / 100) - achat - a.emballage
    prix_max = revente / seuil
    favorable = a.decision.strip().upper().startswith("ACHETER")
    acheter = favorable and mult >= seuil and net > 0 and a.groupe.strip().upper()[:1] != "X"
    couleur = "#1f9d55" if acheter else "#d64545"

    st.caption(f"Article : {md(r['nom'])}")
    st.markdown(
        f'<div class="verdict" style="border-color:{couleur};background:{couleur}22">'
        f'<span class="t" style="color:{couleur}">{"ACHETER" if acheter else "NE PAS ACHETER"}</span>'
        f'<span class="m">Potentiel ×{fr(mult)}<small>seuil ×{fr(seuil)}</small></span></div>',
        unsafe_allow_html=True,
    )
    if favorable and not acheter:
        st.info(f"L'IA était plutôt favorable, mais le calcul ne passe pas : ×{fr(mult)} pour un seuil de "
                f"×{fr(seuil)}, net de {fr(net)} €.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Revente estimée", f"{fr(revente)} €")
    c2.metric("Net par pièce", f"{fr(net)} €")
    c3.metric("Achat max conseillé", f"{fr(prix_max)} €")
    c4, c5, c6 = st.columns(3)
    c4.metric("Fourchette", f"{fr(a.prix_revente_min, 0)}–{fr(a.prix_revente_max, 0)} €")
    c5.metric("Vendu en", f"≈ {a.delai_vente_jours} j")
    c6.metric("Risque", a.risque.capitalize())

    for ligne in a.justification[:3]:
        st.markdown(f"- {md(ligne)}")
    if a.a_verifier:
        st.markdown(f"**À vérifier avant d'acheter :** {md(a.a_verifier)}")

    q = quote_plus(r["nom"])
    l1, l2 = st.columns(2)
    l1.link_button("Annonces Vinted", f"https://www.vinted.fr/catalog?search_text={q}")
    l2.link_button("Ventes conclues eBay", f"https://www.ebay.fr/sch/i.html?_nkw={q}&LH_Sold=1&LH_Complete=1")

    origine = "saisi" if saisi > 0 else "estimé, prix typique en brocante"
    st.caption(
        f"Achat pris en compte : {fr(achat)} € ({origine}). Net = revente × {fr(1 - taux / 100, 3)} − achat − "
        f"emballage ({fr(a.emballage)} €). Catégorie : {md(a.categorie)}. Confiance de l'IA : {a.confiance}. "
        f"Modèle : {r['modele']}. Estimation de l'IA d'après ton guide, pas des ventes réelles : "
        "vérifie au moins 3 annonces comparables avant d'acheter."
    )


# --- Interface ------------------------------------------------------------------
st.markdown(
    """<style>
.verdict{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:.4rem 1.5rem;
padding:1rem 1.25rem;border-radius:.6rem;border-left:.55rem solid;margin:.25rem 0 1rem}
.verdict .t{font-size:2rem;font-weight:800;line-height:1.1}
.verdict .m{font-size:1.5rem;font-weight:700}
.verdict .m small{font-size:.85rem;font-weight:400;opacity:.7;margin-left:.5rem}
</style>""",
    unsafe_allow_html=True,
)

# Accès protégé par mot de passe (optionnel) : évite que d'autres consomment ton quota gratuit.
mot_de_passe = lire_secret("APP_PASSWORD")
if mot_de_passe and not st.session_state.get("acces"):
    saisie = st.text_input("Mot de passe", type="password")
    if saisie and hmac.compare_digest(saisie.encode(), str(mot_de_passe).encode()):
        st.session_state["acces"] = True
        st.rerun()
    elif saisie:
        st.error("Mot de passe incorrect.")
    st.stop()

with st.sidebar:
    st.header("Réglages")
    seuil = st.slider("Multiplicateur minimum", 2.0, 6.0, 4.0, 0.5, help="Le guide recommande ×4 minimum.")
    taux = st.number_input("Cotisations (%)", 0.0, 30.0, 12.3, 0.1,
                           help="12,3 % en micro-entreprise (vente de marchandises). Mets 0 pour ne pas les déduire.")

st.title("Analyseur d'achat Vinted")
st.write("Décris l'article, ajoute une photo : l'IA applique ton guide et tranche.")

nom = st.text_input("Nom exact de l'article", placeholder="Ex. Patagonia Better Sweater 1/4 zip homme M gris")
nature = st.text_area("Nature de l'article", height=80,
                      placeholder="Catégorie, marque, taille, matière, défauts… (ex. polaire homme, taille M)")
col_etat, col_prix = st.columns(2)
etat = col_etat.selectbox("État", ETATS, index=3)
prix = col_prix.number_input("Prix demandé (€), optionnel", min_value=0.0, step=0.5, format="%.2f",
                             help="Laisse 0 si tu ne connais pas encore le prix : l'app te donne le prix d'achat maximum.")

mode = st.radio("Photo", ["Importer une photo", "Prendre une photo"], horizontal=True)
if mode == "Importer une photo":
    photo = st.file_uploader("Photo de l'article", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if photo:
        st.image(photo, width=220)
else:
    photo = st.camera_input("Photo de l'article", label_visibility="collapsed")

if st.button("Analyser l'article", type="primary"):
    cle = lire_secret("GEMINI_API_KEY") or lire_secret("GOOGLE_API_KEY")
    if not nom.strip():
        st.warning("Indique le nom exact de l'article.")
    elif not cle:
        st.error("Clé API manquante : définis GEMINI_API_KEY (variable d'environnement ou Secrets Streamlit).")
    else:
        with st.spinner("Analyse en cours…"):
            try:
                jpeg = preparer_image(photo) if photo else None
                demande = construire_demande(nom.strip(), nature.strip(), etat, prix, seuil, bool(jpeg))
                analyse, modele = analyser(cle, demande, jpeg)
                st.session_state["resultat"] = dict(
                    nom=nom.strip(), a=analyse, saisi=prix, seuil=seuil, taux=taux, modele=modele
                )
            except Exception as e:
                st.session_state.pop("resultat", None)
                st.error(expliquer(e))

if "resultat" in st.session_state:
    st.divider()
    afficher_resultat(st.session_state["resultat"])
