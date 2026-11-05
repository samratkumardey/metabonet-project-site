/* ---------------------------------------------------------------------------
   SITE CONFIG — the only file you need to edit to de-anonymise the page.

   While the work is under double-blind review keep `anonymous: true`.
   After acceptance set it to false and fill in the `named` block; the hero,
   the BibTeX entry, the footer and the acknowledgements all read from here.
   --------------------------------------------------------------------------- */
window.SITE = {
  anonymous: true,

  // Shown while anonymous is true
  anon: {
    authors: [
      { name: "Anonymous Author", aff: [1] },
      { name: "Anonymous Author", aff: [1] },
    ],
    affiliations: ["Anonymous Institution"],
    team: "withheld for review",
    acknowledgements:
      "Acknowledgements are withheld for anonymous review.",
    bibAuthor: "Anonymous",
  },

  // Filled in after review — edit these, then set anonymous: false
  named: {
    authors: [
      { name: "First Author", aff: [1] },
      { name: "Senior Author", aff: [1] },
    ],
    affiliations: ["Department, University"],
    team: "team name on the MetaboNet leaderboard",
    acknowledgements:
      "Computation for this work was performed on the high-performance computing " +
      "infrastructure provided by <institution>.",
    bibAuthor: "Last, First and Last, First",
  },

  // Links in the hero badge row. Leave "#" to show the badge as "coming soon".
  links: {
    paper: "#",
    code: "#",
    leaderboard: "https://metabonetglucose-leaderboard.hf.space/",
    toolkit: "https://github.com/replicahealth/glucose-prediction-submission-toolkit",
    dataset: "https://metabo-net.org/",
  },

  bib: {
    key: "MetaboNetCGM2026",
    title:
      "Information, Not Architecture: Forecasting Glucose 30 to 120 Minutes Ahead " +
      "on the MetaboNet Benchmark",
    venue: "arXiv preprint",
    year: "2026",
    url: "#",
  },
};
