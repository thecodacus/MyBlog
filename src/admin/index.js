import CMS from "netlify-cms-app"
import BlogPostPreview from "./preview-templates/posts-details-previews"

CMS.registerPreviewTemplate("blog", BlogPostPreview)
