# Adding Custom Roles to the Resume Screening System

The role database (`role_database.csv`) controls how the system classifies resumes. You can easily add, modify, or remove roles.

## CSV Structure

Each row in `role_database.csv` represents one role with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `role_name` | Name of the role | "DevOps Engineer" |
| `keywords` | Comma-separated keywords to match | "devops,ci/cd,jenkins,docker,kubernetes" |
| `certifications` | Relevant certifications | "AWS Certified DevOps Engineer,CKA" |
| `experience_level` | Experience level | "junior-senior" or "mid-senior" |
| `description` | Brief description | "Automates deployment pipelines..." |
| `salary_range` | Typical salary range | "90k-170k" |
| `required_skills` | Key skills needed | "CI/CD, Docker, Kubernetes, Linux" |

## Adding a New Role

1. Open `role_database.csv` in Excel, Google Sheets, or a text editor
2. Add a new row with your role details
3. Keywords should be lowercase and comma-separated (no spaces after commas)
4. Save the file

**Example - Adding "AI Research Scientist":**

```csv
AI Research Scientist,"artificial intelligence,research,deep learning,nlp,computer vision,pytorch,tensorflow,publications","PhD in Computer Science,Published Research Papers",senior,"Conducts cutting-edge AI research and publishes papers","120k-250k","Deep Learning, Research methodology, PyTorch/TensorFlow, Mathematics, Publications"
```

## Updating Keywords for Existing Roles

If the system isn't detecting a role correctly, add more keywords:

**Before:**
```csv
Frontend Developer,"html,css,javascript,react",...
```

**After (more comprehensive):**
```csv
Frontend Developer,"html,css,javascript,typescript,react,vue,angular,next.js,tailwind,webpack,sass",...
```

## Testing Your Changes

After editing `role_database.csv`, test immediately:

```bash
# Test basic detection
python predict_cli.py "keywords from your new role..."

# See full details
python predict_cli.py -v "keywords from your new role..."
```

## Tips for Good Keywords

1. **Use lowercase** - all matching is case-insensitive
2. **Include variations** - add both "machine learning" and "ml"
3. **Add specific technologies** - framework names, tools, languages
4. **Keep it focused** - keywords should be unique to this role
5. **Test frequently** - add a keyword, test, repeat

## Common Roles to Add

Here are suggestions for additional roles you might want to add:

- **Technical Writer** - documentation, technical writing, APIs
- **Solutions Architect** - architecture, system design, scalability
- **Network Engineer** - networking, cisco, routing, switching
- **Business Analyst** - requirements, stakeholders, analysis
- **Scrum Master** - agile, scrum, sprint planning, ceremonies
- **MLOps Engineer** - mlops, model deployment, kubeflow, mlflow
