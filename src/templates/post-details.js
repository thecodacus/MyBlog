import React from "react"
import { graphql } from "gatsby"
import { Disqus } from "gatsby-plugin-disqus"
import Layout from "../components/Layout"
import Seo from "../components/Seo"
import Article from "../components/Article"

export default function PostDetails({ data }) {
	// console.log(data)
	const props = {
		...data,
		...data?.markdownRemark,
		...data?.markdownRemark?.frontmatter,
		...data?.markdownRemark?.frontmatter?.featuredImage,
	}

	let disqusConfig = {
		// url: `${data.site.siteMetadata.url}`,
		identifier: props.id,
		title: props.title,
	}

	return (
		<Layout>
			<Seo title={props.title} description={props.excerpt} />
			<Article {...props}>
				<Disqus config={disqusConfig} />
			</Article>
		</Layout>
	)
}

export const query = graphql`
	query PostDetails($slug: String) {
		site {
			siteMetadata {
				url
			}
		}
		markdownRemark(fields: { slug: { eq: $slug } }) {
			id
			html
			fields {
				slug
			}
			excerpt
			frontmatter {
				date
				slug
				category
				title
				featuredImage {
					publicURL
					childImageSharp {
						gatsbyImageData(layout: FULL_WIDTH)
					}
				}
			}
		}
	}
`
