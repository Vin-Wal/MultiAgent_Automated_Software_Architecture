"""
Three evaluation scenarios covering different domains and compliance profiles.
Used by run_eval.py for both pipeline runs and result labelling.
"""

SCENARIOS = [
    {
        "name": "ride_sharing",
        "label": "Ride-Sharing",
        "input": (
            "Build a real-time ride-sharing platform for 500k concurrent users. "
            "Features: rider/driver matching in under 3 seconds, live GPS tracking, "
            "dynamic surge pricing, payment processing (cards + wallets), trip history, "
            "and driver earnings dashboard. "
            "NFRs: 99.99% uptime, p99 match latency < 3s, PCI-DSS for payments, "
            "GDPR compliance, multi-region active-active deployment."
        ),
    },
    {
        "name": "healthcare_records",
        "label": "Healthcare EHR",
        "input": (
            "Build a cloud-based electronic health records system for a network of 200 clinics. "
            "Features: patient record management, appointment scheduling, prescription management, "
            "lab results integration, and a patient portal for self-service access. "
            "NFRs: HIPAA compliance, 99.9% uptime, sub-500ms page loads, "
            "audit logging for all record access, role-based access for doctors/nurses/admins/patients, "
            "end-to-end encryption at rest and in transit."
        ),
    },
    {
        "name": "ecommerce",
        "label": "E-Commerce",
        "input": (
            "Build a multi-vendor e-commerce marketplace supporting 10,000 sellers and 1M monthly buyers. "
            "Features: product catalog with full-text search and filtering, shopping cart, "
            "checkout with Stripe, order tracking, seller dashboard with analytics, "
            "and a review and rating system. "
            "NFRs: 99.9% uptime, search results in under 200ms, PCI-DSS compliance, "
            "support for 50,000 concurrent users during flash sales."
        ),
    },
]
