const featuresData = {
  "Accessibility": [
    "Access for people with reduced mobility",
    "Handicap access"
  ],
  "Security": [
    "Alarm System",
    "Domotics",
    "Electric Blinds",
    "Entry Phone",
    "Gated Complex",
    "Safe",
    "Satellite TV"
  ],
  "Location and Proximity": [
    "Beachfront",
    "Beachside",
    "Close To Forest",
    "Close To Marina",
    "Close To Sea",
    "Close To Skiing",
    "Close to Golf",
    "Close to Schools",
    "Close to Shops",
    "Close to Town",
    "Close to port",
    "Near Church",
    "Near Mosque",
    "Near Transport",
    "Restaurant On Site",
    "Commercial Area",
    "Car Hire Facility",
    "Courtesy Bus",
    "Day Care",
    "Suburban",
    "Village",
    "Urbanisation",
    "Port",
    "Marina",
    "Town"
  ],
  "Climate Control": [
    "Air Conditioning",
    "Cold A/C",
    "Hot A/C",
    "Pre Installed A/C",
    "Fireplace",
    "U/F Heating",
    "Central Heating",
    "Climate Control"
  ],
  "Parking": [
    "Communal",
    "Covered",
    "Garage",
    "Underground",
    "Street",
    "Private",
    "Open",
    "More Than One"
  ],
  "Condition": [
    "Excellent",
    "Good",
    "Fair",
    "Recently Refurbished",
    "Recently Renovated",
    "New Construction",
    "Renovation Required",
    "Restoration Required"
  ],
  "Views": [
    "Beach",
    "Courtyard",
    "Country",
    "Forest",
    "Garden",
    "Golf",
    "Lake",
    "Mountain",
    "Panoramic",
    "Pool",
    "Port",
    "Sea",
    "Ski",
    "Street",
    "Urban"
  ],
  "Orientation": [
    "North",
    "North East",
    "North West",
    "East",
    "West",
    "South East",
    "South",
    "South West"
  ],
  "Kitchen": [
    "Not Fitted",
    "Partially Fitted",
    "Fully Fitted",
    "Kitchen-Lounge"
  ],
  "Utilities and Connectivity": [
    "Electricity",
    "Gas",
    "Drinkable Water",
    "Telephone",
    "Fiber Optic",
    "WiFi",
    "Photovoltaic solar panels",
    "Solar water heating"
  ],
  "Furnishing": [
    "Fully Furnished",
    "Not Furnished",
    "Part Furnished",
    "Optional Furniture"
  ],
  "Lifestyle and Amenities": [
    "Bar",
    "Barbeque",
    "Bargain",
    "Basement",
    "Cheap",
    "Communal Garden",
    "Communal Pool",
    "Children's Pool",
    "Covered Terrace",
    "Distressed",
    "Double Glazing",
    "Easy Maintenance Garden",
    "Ensuite Bathroom",
    "Fitted Wardrobes",
    "Front Line Beach Complex",
    "Frontline Golf",
    "Games Room",
    "Golf",
    "Guest Apartment",
    "Guest House",
    "Gym",
    "Heated Pool",
    "Holiday Homes",
    "Indoor Pool",
    "Investment",
    "Jacuzzi",
    "Landscaped Garden",
    "Lift",
    "Luxury",
    "Marble Flooring",
    "Mountain Pueblo",
    "Off Plan",
    "Paddle Tennis",
    "Private Garden",
    "Private Pool",
    "Private Terrace",
    "Reduced",
    "Repossession",
    "Resale",
    "Room For Pool",
    "Sauna",
    "Solarium",
    "Stables",
    "Staff Accommodation",
    "Storage Room",
    "Tennis Court",
    "Utility Room",
    "With Planning Permission",
    "Wood Flooring",
    "Contemporary",
    "Country"
  ]
};

function featureToId(featureName) {
  return featureName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '')
    .replace(/^([0-9])/, '_$1');
}

function categoryToId(categoryName) {
  return categoryName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '')
    .replace(/^([0-9])/, '_$1') + 'Features';
}

function createFeatureCheckbox(featureName, categoryId) {
  const id = featureToId(featureName);
  return `
    <div class="form-check">
      <input type="checkbox" class="form-check-input" id="${id}" name="features" value="${featureName}">
      <label class="form-check-label" for="${id}">${featureName}</label>
    </div>
  `;
}

function createFeatureCategory(categoryName, features) {
  const categoryId = categoryToId(categoryName);
  const featuresHtml = features.map(feature => createFeatureCheckbox(feature, categoryId)).join('');
  
  return `
    <div class="mb-3">
      <button class="btn btn-outline-secondary w-100" type="button" data-bs-toggle="collapse" data-bs-target="#${categoryId}" aria-expanded="false" aria-controls="${categoryId}">
        ${categoryName}
      </button>
      <div class="collapse mt-2" id="${categoryId}">
        ${featuresHtml}
      </div>
    </div>`;
}

function generateFeatureDropdowns() {
  const container = document.getElementById('featuresContainer');
  if (!container) {
    console.error('Features container not found!');
    return;
  }
  
  let html = '';
  for (const [categoryName, features] of Object.entries(featuresData)) {
    html += createFeatureCategory(categoryName, features);
  }
  
  container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', generateFeatureDropdowns);
