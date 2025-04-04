"""
Module containing information about different types of brain tumors.
This provides educational information to be shown after a diagnosis.
"""

# Dictionary of tumor information
TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma',
        'description': 'Gliomas are tumors that grow in the brain or spine. They arise from glial cells, which provide support and protection for neurons in the brain.',
        'types': [
            'Astrocytoma - forms in astrocytes, star-shaped cells',
            'Oligodendroglioma - develops from oligodendrocytes',
            'Ependymoma - forms from ependymal cells lining the ventricles'
        ],
        'symptoms': [
            'Headaches that may be more severe in the morning',
            'Seizures',
            'Personality changes',
            'Nausea and vomiting',
            'Changes in vision, hearing, or speech',
            'Loss of balance'
        ],
        'treatment_options': [
            'Surgery - to remove as much of the tumor as possible',
            'Radiation therapy - to kill remaining tumor cells',
            'Chemotherapy - drugs to kill cancer cells',
            'Targeted drug therapy - focuses on specific abnormalities in cancer cells'
        ],
        'prognosis': 'Prognosis varies greatly depending on the type, grade, location, age, and overall health. Low-grade gliomas generally have a better prognosis than high-grade ones.',
        'risk_factors': [
            'Exposure to radiation',
            'Family history of gliomas',
            'Genetic disorders such as neurofibromatosis and Li-Fraumeni syndrome'
        ],
        'sources': [
            'https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251',
            'https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq'
        ]
    },
    
    'meningioma': {
        'name': 'Meningioma',
        'description': 'Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign (not cancerous).',
        'types': [
            'Grade I (benign) - slow-growing and non-cancerous',
            'Grade II (atypical) - more likely to recur than Grade I',
            'Grade III (anaplastic/malignant) - aggressive and may invade brain tissue'
        ],
        'symptoms': [
            'Headaches that worsen over time',
            'Seizures',
            'Changes in vision',
            'Weakness in arms or legs',
            'Language difficulty',
            'Memory issues'
        ],
        'treatment_options': [
            'Observation - for small, slow-growing tumors without symptoms',
            'Surgery - to remove the tumor if possible',
            'Radiation therapy - particularly for tumors that cannot be fully removed',
            'Radiosurgery - precise radiation delivery to the tumor'
        ],
        'prognosis': 'Most meningiomas are benign and grow slowly. The outlook is usually favorable, especially for Grade I tumors that can be completely removed. However, even benign meningiomas can recur.',
        'risk_factors': [
            'Previous radiation therapy to the head',
            'Female hormones (more common in women)',
            'Neurofibromatosis type 2 genetic disorder',
            'Age (more common in older adults)'
        ],
        'sources': [
            'https://www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643',
            'https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults/about/meningioma.html'
        ]
    },
    
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'Pituitary tumors develop in the pituitary gland at the base of the brain. They can affect hormone production and are usually benign.',
        'types': [
            'Functioning tumors - produce hormones',
            'Non-functioning tumors - do not produce hormones',
            'Microadenomas - smaller than 10mm',
            'Macroadenomas - larger than 10mm'
        ],
        'symptoms': [
            'Headaches',
            'Vision problems',
            'Hormonal imbalances causing various symptoms',
            'Cushing\'s syndrome (excess cortisol)',
            'Acromegaly (excess growth hormone)',
            'Hyperthyroidism (excess thyroid hormone)'
        ],
        'treatment_options': [
            'Medication - to reduce tumor size or normalize hormone production',
            'Surgery - typically transsphenoidal (through the nose)',
            'Radiation therapy - for tumors that recur or cannot be completely removed',
            'Hormone replacement - if the tumor affects normal hormone production'
        ],
        'prognosis': 'Most pituitary tumors are benign and have a good prognosis. Success rates depend on tumor size, type, and whether it can be completely removed.',
        'risk_factors': [
            'Family history of pituitary tumors',
            'Multiple endocrine neoplasia type 1 (MEN1)',
            'Carney complex',
            'McCune-Albright syndrome'
        ],
        'sources': [
            'https://www.mayoclinic.org/diseases-conditions/pituitary-tumors/symptoms-causes/syc-20350548',
            'https://www.cancer.org/cancer/types/pituitary-tumors.html'
        ]
    },
    
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'No evidence of a brain tumor was found in the scan. However, this does not rule out other neurological conditions.',
        'types': [],
        'symptoms': [
            'If you are experiencing neurological symptoms but no tumor was detected, further evaluation may be needed to identify other possible causes.'
        ],
        'treatment_options': [
            'Follow up with your healthcare provider to discuss your symptoms',
            'Additional diagnostic tests may be recommended'
        ],
        'prognosis': 'Not applicable as no tumor was detected.',
        'risk_factors': [],
        'sources': [
            'Consult with your healthcare provider for personalized medical advice.'
        ]
    }
}

def get_tumor_info(tumor_type):
    """
    Get information about a specific tumor type.
    
    Args:
        tumor_type (str): The type of tumor (e.g., 'glioma', 'meningioma', 'pituitary')
        
    Returns:
        dict: Information about the tumor type, or None if not found
    """
    return TUMOR_INFO.get(tumor_type.lower())

def get_all_tumor_types():
    """
    Get a list of all available tumor types.
    
    Returns:
        list: List of tumor type names
    """
    return list(TUMOR_INFO.keys()) 