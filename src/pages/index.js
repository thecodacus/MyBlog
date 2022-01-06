// import { graphql } from "gatsby"
import { graphql } from "gatsby"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import React from "react"
import Layout from "../components/Layout"
import PostsGrid from "../components/PostsGrid"
import Seo from "../components/Seo"
import * as styles from "../styles/home.module.scss"

export default function Home({ data }) {
	console.log(data)
	const { banner, info, posts } = data
	return (
		<Layout>
			<Seo title="Home" />
			<div className={styles.home}>
				<section className={styles.hero} style={{ filter: "drop-shadow(7px 8px 5px rgb(0, 0, 0, 0.3))" }}>
					<div className={styles.heroText}>
						<div className={styles.description}>Welcome To {info.siteMetadata.subHeader}</div>
						<div className={styles.title}>{info.siteMetadata.description}</div>
						<div className={styles.ps}>{info.siteMetadata.ps}</div>
					</div>
					{banner.childImageSharp != null ? (
						<GatsbyImage image={getImage(banner)} alt="banner image" />
					) : (
						<img
							style={{
								width: "calc(100% - 2rem)",
								padding: "1rem",
								maxWidth: "600px",
								marginLeft: "auto",
								// filter: "drop-shadow(7px 8px 5px rgb(0, 0, 0, 0.3))",
							}}
							src={banner.publicURL}
							alt="banner image"
						/>
					)}
				</section>
				<section className="postgrid">
					<PostsGrid posts={posts.nodes} />
				</section>
			</div>
		</Layout>
	)
}
export const query = graphql`
	query HomePageContent {
		banner: file(relativePath: { eq: "homepage-banner.svg" }) {
			publicURL
			childImageSharp {
				gatsbyImageData(layout: FULL_WIDTH)
			}
		}
		info: site {
			siteMetadata {
				title
				subHeader
				description
				ps
			}
		}
		posts: allMarkdownRemark(sort: { fields: frontmatter___date, order: DESC }) {
			nodes {
				fields {
					slug
				}
				excerpt
				frontmatter {
					title
					category
					date
					featuredImage {
						publicURL
						childImageSharp {
							gatsbyImageData(aspectRatio: 1.5, layout: FULL_WIDTH)
						}
					}
				}
				id
			}
		}
	}
`
