// Krakey i18n core — mirrors the theme-toggle localStorage pattern.
// A locale key is stored in 'krakey-lang'; on load the stored value
// (or 'en') becomes the active locale. setLocale() swaps the active
// locale and persists the choice; the app.js IIFE is responsible for
// updating <html lang> and re-rendering strings via applyLocale().
(function () {
  window.LOCALES = { en: {} };   // en starts empty; later units inject keys

  var _locale = localStorage.getItem('krakey-lang') || 'en';

  window.getLocale = function () { return _locale; };

  window.availableLocales = function () { return Object.keys(window.LOCALES); };

  window.t = function (key) {
    return (window.LOCALES[_locale] && window.LOCALES[_locale][key]) ||
           (window.LOCALES.en     && window.LOCALES.en[key])         ||
           key;
  };

  // NO-OP for unregistered langs. Does NOT touch <html lang> or any
  // button — that responsibility belongs to the app.js IIFE.
  window.setLocale = function (lang) {
    if (!window.LOCALES[lang]) return;
    _locale = lang;
    localStorage.setItem('krakey-lang', lang);
  };
})();
