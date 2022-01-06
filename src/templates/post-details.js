import React from "react"
import { graphql } from "gatsby"
import { Disqus, CommentCount } from "gatsby-plugin-disqus"
import Layout from "../components/Layout"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import * as styles from "../styles/post-details.module.scss"
import Seo from "../components/Seo"

export default function PostDetails({ data }) {
	console.log(data)
	const { id, html, excerpt } = data.markdownRemark
	// const { slug } = data.markdownRemark.fields
	const { title, category, featuredImage, date } = data.markdownRemark.frontmatter
	const featuredImagePath = featuredImage.publicURL

	let disqusConfig = {
		// url: `${data.site.siteMetadata.url}`,
		identifier: id,
		title: title,
	}

	return (
		<Layout>
			<Seo title={title} description={excerpt} />
			<div className={styles.details}>
				<h1>{title}</h1>
				<div className={styles.metadata}>
					<h3>{new Date(date).toDateString()}</h3>
					<h3>.</h3>
					<h3>{category}</h3>
					<CommentCount config={disqusConfig} placeholder={"..."} />
				</div>

				<div className={styles.featured}>
					{featuredImage.childImageSharp != null ? (
						<GatsbyImage image={getImage(featuredImage)} alt={title} />
					) : (
						<img className={styles.alternateImage} src={featuredImagePath} alt={title} />
					)}
				</div>
				<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>
				<Disqus config={disqusConfig} />
			</div>
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
